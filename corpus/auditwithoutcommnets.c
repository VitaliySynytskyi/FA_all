#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/file.h>
#include <linux/init.h>
#include <linux/types.h>
#include <linux/atomic.h>
#include <linux/mm.h>
#include <linux/export.h>
#include <linux/slab.h>
#include <linux/err.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/syscalls.h>

#include <linux/audit.h>

#include <net/sock.h>
#include <net/netlink.h>
#include <linux/skbuff.h>
#ifdef CONFIG_SECURITY
#include <linux/security.h>
#endif
#include <linux/freezer.h>
#include <linux/tty.h>
#include <linux/pid_namespace.h>
#include <net/netns/generic.h>

#include "audit.h"

#define AUDIT_DISABLED		-1
#define AUDIT_UNINITIALIZED	0
#define AUDIT_INITIALIZED	1
static int	audit_initialized;

#define AUDIT_OFF	0
#define AUDIT_ON	1
#define AUDIT_LOCKED	2
u32		audit_enabled;
u32		audit_ever_enabled;

EXPORT_SYMBOL_GPL(audit_enabled);

static u32	audit_default;

static u32	audit_failure = AUDIT_FAIL_PRINTK;

int		audit_pid;
static __u32	audit_nlk_portid;

static u32	audit_rate_limit;

static u32	audit_backlog_limit = 64;
#define AUDIT_BACKLOG_WAIT_TIME (60 * HZ)
static u32	audit_backlog_wait_time_master = AUDIT_BACKLOG_WAIT_TIME;
static u32	audit_backlog_wait_time = AUDIT_BACKLOG_WAIT_TIME;

kuid_t		audit_sig_uid = INVALID_UID;
pid_t		audit_sig_pid = -1;
u32		audit_sig_sid = 0;

static atomic_t    audit_lost = ATOMIC_INIT(0);

static struct sock *audit_sock;
static int audit_net_id;

struct list_head audit_inode_hash[AUDIT_INODE_BUCKETS];

static DEFINE_SPINLOCK(audit_freelist_lock);
static int	   audit_freelist_count;
static LIST_HEAD(audit_freelist);

static struct sk_buff_head audit_skb_queue;
static struct sk_buff_head audit_skb_hold_queue;
static struct task_struct *kauditd_task;
static DECLARE_WAIT_QUEUE_HEAD(kauditd_wait);
static DECLARE_WAIT_QUEUE_HEAD(audit_backlog_wait);

static struct audit_features af = {.vers = AUDIT_FEATURE_VERSION,
				   .mask = -1,
				   .features = 0,
				   .lock = 0,};

static char *audit_feature_names[2] = {
	"only_unset_loginuid",
	"loginuid_immutable",
};


DEFINE_MUTEX(audit_cmd_mutex);

#define AUDIT_BUFSIZ 1024

#define AUDIT_MAXFREE  (2*NR_CPUS)

struct audit_buffer {
	struct list_head     list;
	struct sk_buff       *skb;
	struct audit_context *ctx;	
	gfp_t		     gfp_mask;
};

struct audit_reply {
	__u32 portid;
	struct net *net;
	struct sk_buff *skb;
};

static void audit_set_portid(struct audit_buffer *ab, __u32 portid)
{
	if (ab) {
		struct nlmsghdr *nlh = nlmsg_hdr(ab->skb);
		nlh->nlmsg_pid = portid;
	}
}

void audit_panic(const char *message)
{
	switch (audit_failure) {
	case AUDIT_FAIL_SILENT:
		break;
	case AUDIT_FAIL_PRINTK:
		if (printk_ratelimit())
			pr_err("%s\n", message);
		break;
	case AUDIT_FAIL_PANIC:
		if (audit_pid)
			panic("audit: %s\n", message);
		break;
	}
}

static inline int audit_rate_check(void)
{
	static unsigned long	last_check = 0;
	static int		messages   = 0;
	static DEFINE_SPINLOCK(lock);
	unsigned long		flags;
	unsigned long		now;
	unsigned long		elapsed;
	int			retval	   = 0;

	if (!audit_rate_limit) return 1;

	spin_lock_irqsave(&lock, flags);
	if (++messages < audit_rate_limit) {
		retval = 1;
	} else {
		now     = jiffies;
		elapsed = now - last_check;
		if (elapsed > HZ) {
			last_check = now;
			messages   = 0;
			retval     = 1;
		}
	}
	spin_unlock_irqrestore(&lock, flags);

	return retval;
}

void audit_log_lost(const char *message)
{
	static unsigned long	last_msg = 0;
	static DEFINE_SPINLOCK(lock);
	unsigned long		flags;
	unsigned long		now;
	int			print;

	atomic_inc(&audit_lost);

	print = (audit_failure == AUDIT_FAIL_PANIC || !audit_rate_limit);

	if (!print) {
		spin_lock_irqsave(&lock, flags);
		now = jiffies;
		if (now - last_msg > HZ) {
			print = 1;
			last_msg = now;
		}
		spin_unlock_irqrestore(&lock, flags);
	}

	if (print) {
		if (printk_ratelimit())
			pr_warn("audit_lost=%u audit_rate_limit=%u audit_backlog_limit=%u\n",
				atomic_read(&audit_lost),
				audit_rate_limit,
				audit_backlog_limit);
		audit_panic(message);
	}
}

static int audit_log_config_change(char *function_name, u32 new, u32 old,
				   int allow_changes)
{
	struct audit_buffer *ab;
	int rc = 0;

	ab = audit_log_start(NULL, GFP_KERNEL, AUDIT_CONFIG_CHANGE);
	if (unlikely(!ab))
		return rc;
	audit_log_format(ab, "%s=%u old=%u", function_name, new, old);
	audit_log_session_info(ab);
	rc = audit_log_task_context(ab);
	if (rc)
		allow_changes = 0;
	audit_log_format(ab, " res=%d", allow_changes);
	audit_log_end(ab);
	return rc;
}

static int audit_do_config_change(char *function_name, u32 *to_change, u32 new)
{
	int allow_changes, rc = 0;
	u32 old = *to_change;

	if (audit_enabled == AUDIT_LOCKED)
		allow_changes = 0;
	else
		allow_changes = 1;

	if (audit_enabled != AUDIT_OFF) {
		rc = audit_log_config_change(function_name, new, old, allow_changes);
		if (rc)
			allow_changes = 0;
	}

	if (allow_changes == 1)
		*to_change = new;
	else if (rc == 0)
		rc = -EPERM;
	return rc;
}

static int audit_set_rate_limit(u32 limit)
{
	return audit_do_config_change("audit_rate_limit", &audit_rate_limit, limit);
}

static int audit_set_backlog_limit(u32 limit)
{
	return audit_do_config_change("audit_backlog_limit", &audit_backlog_limit, limit);
}

static int audit_set_backlog_wait_time(u32 timeout)
{
	return audit_do_config_change("audit_backlog_wait_time",
				      &audit_backlog_wait_time_master, timeout);
}

static int audit_set_enabled(u32 state)
{
	int rc;
	if (state > AUDIT_LOCKED)
		return -EINVAL;

	rc =  audit_do_config_change("audit_enabled", &audit_enabled, state);
	if (!rc)
		audit_ever_enabled |= !!state;

	return rc;
}

static int audit_set_failure(u32 state)
{
	if (state != AUDIT_FAIL_SILENT
	    && state != AUDIT_FAIL_PRINTK
	    && state != AUDIT_FAIL_PANIC)
		return -EINVAL;

	return audit_do_config_change("audit_failure", &audit_failure, state);
}

static void audit_hold_skb(struct sk_buff *skb)
{
	if (audit_default &&
	    (!audit_backlog_limit ||
	     skb_queue_len(&audit_skb_hold_queue) < audit_backlog_limit))
		skb_queue_tail(&audit_skb_hold_queue, skb);
	else
		kfree_skb(skb);
}

static void audit_printk_skb(struct sk_buff *skb)
{
	struct nlmsghdr *nlh = nlmsg_hdr(skb);
	char *data = nlmsg_data(nlh);

	if (nlh->nlmsg_type != AUDIT_EOE) {
		if (printk_ratelimit())
			pr_notice("type=%d %s\n", nlh->nlmsg_type, data);
		else
			audit_log_lost("printk limit exceeded");
	}

	audit_hold_skb(skb);
}

static void kauditd_send_skb(struct sk_buff *skb)
{
	int err;
	int attempts = 0;
#define AUDITD_RETRIES 5

restart:
	skb_get(skb);
	err = netlink_unicast(audit_sock, skb, audit_nlk_portid, 0);
	if (err < 0) {
		pr_err("netlink_unicast sending to audit_pid=%d returned error: %d\n",
		       audit_pid, err);
		if (audit_pid) {
			if (err == -ECONNREFUSED || err == -EPERM
			    || ++attempts >= AUDITD_RETRIES) {
				char s[32];

				snprintf(s, sizeof(s), "audit_pid=%d reset", audit_pid);
				audit_log_lost(s);
				audit_pid = 0;
				audit_sock = NULL;
			} else {
				pr_warn("re-scheduling(#%d) write to audit_pid=%d\n",
					attempts, audit_pid);
				set_current_state(TASK_INTERRUPTIBLE);
				schedule();
				__set_current_state(TASK_RUNNING);
				goto restart;
			}
		}
		audit_hold_skb(skb);
	} else
		consume_skb(skb);
}

static void kauditd_send_multicast_skb(struct sk_buff *skb, gfp_t gfp_mask)
{
	struct sk_buff		*copy;
	struct audit_net	*aunet = net_generic(&init_net, audit_net_id);
	struct sock		*sock = aunet->nlsk;

	if (!netlink_has_listeners(sock, AUDIT_NLGRP_READLOG))
		return;

	copy = skb_copy(skb, gfp_mask);
	if (!copy)
		return;

	nlmsg_multicast(sock, copy, 0, AUDIT_NLGRP_READLOG, gfp_mask);
}

static void flush_hold_queue(void)
{
	struct sk_buff *skb;

	if (!audit_default || !audit_pid)
		return;

	skb = skb_dequeue(&audit_skb_hold_queue);
	if (likely(!skb))
		return;

	while (skb && audit_pid) {
		kauditd_send_skb(skb);
		skb = skb_dequeue(&audit_skb_hold_queue);
	}

	consume_skb(skb);
}

static int kauditd_thread(void *dummy)
{
	set_freezable();
	while (!kthread_should_stop()) {
		struct sk_buff *skb;

		flush_hold_queue();

		skb = skb_dequeue(&audit_skb_queue);

		if (skb) {
			if (!audit_backlog_limit ||
			    (skb_queue_len(&audit_skb_queue) <= audit_backlog_limit))
				wake_up(&audit_backlog_wait);
			if (audit_pid)
				kauditd_send_skb(skb);
			else
				audit_printk_skb(skb);
			continue;
		}

		wait_event_freezable(kauditd_wait, skb_queue_len(&audit_skb_queue));
	}
	return 0;
}

int audit_send_list(void *_dest)
{
	struct audit_netlink_list *dest = _dest;
	struct sk_buff *skb;
	struct net *net = dest->net;
	struct audit_net *aunet = net_generic(net, audit_net_id);

	mutex_lock(&audit_cmd_mutex);
	mutex_unlock(&audit_cmd_mutex);

	while ((skb = __skb_dequeue(&dest->q)) != NULL)
		netlink_unicast(aunet->nlsk, skb, dest->portid, 0);

	put_net(net);
	kfree(dest);

	return 0;
}

struct sk_buff *audit_make_reply(__u32 portid, int seq, int type, int done,
				 int multi, const void *payload, int size)
{
	struct sk_buff	*skb;
	struct nlmsghdr	*nlh;
	void		*data;
	int		flags = multi ? NLM_F_MULTI : 0;
	int		t     = done  ? NLMSG_DONE  : type;

	skb = nlmsg_new(size, GFP_KERNEL);
	if (!skb)
		return NULL;

	nlh	= nlmsg_put(skb, portid, seq, t, size, flags);
	if (!nlh)
		goto out_kfree_skb;
	data = nlmsg_data(nlh);
	memcpy(data, payload, size);
	return skb;

out_kfree_skb:
	kfree_skb(skb);
	return NULL;
}

static int audit_send_reply_thread(void *arg)
{
	struct audit_reply *reply = (struct audit_reply *)arg;
	struct net *net = reply->net;
	struct audit_net *aunet = net_generic(net, audit_net_id);

	mutex_lock(&audit_cmd_mutex);
	mutex_unlock(&audit_cmd_mutex);

	netlink_unicast(aunet->nlsk , reply->skb, reply->portid, 0);
	put_net(net);
	kfree(reply);
	return 0;
}

static void audit_send_reply(struct sk_buff *request_skb, int seq, int type, int done,
			     int multi, const void *payload, int size)
{
	u32 portid = NETLINK_CB(request_skb).portid;
	struct net *net = sock_net(NETLINK_CB(request_skb).sk);
	struct sk_buff *skb;
	struct task_struct *tsk;
	struct audit_reply *reply = kmalloc(sizeof(struct audit_reply),
					    GFP_KERNEL);

	if (!reply)
		return;

	skb = audit_make_reply(portid, seq, type, done, multi, payload, size);
	if (!skb)
		goto out;

	reply->net = get_net(net);
	reply->portid = portid;
	reply->skb = skb;

	tsk = kthread_run(audit_send_reply_thread, reply, "audit_send_reply");
	if (!IS_ERR(tsk))
		return;
	kfree_skb(skb);
out:
	kfree(reply);
}

static int audit_netlink_ok(struct sk_buff *skb, u16 msg_type)
{
	int err = 0;
	if (current_user_ns() != &init_user_ns)
		return -ECONNREFUSED;

	switch (msg_type) {
	case AUDIT_LIST:
	case AUDIT_ADD:
	case AUDIT_DEL:
		return -EOPNOTSUPP;
	case AUDIT_GET:
	case AUDIT_SET:
	case AUDIT_GET_FEATURE:
	case AUDIT_SET_FEATURE:
	case AUDIT_LIST_RULES:
	case AUDIT_ADD_RULE:
	case AUDIT_DEL_RULE:
	case AUDIT_SIGNAL_INFO:
	case AUDIT_TTY_GET:
	case AUDIT_TTY_SET:
	case AUDIT_TRIM:
	case AUDIT_MAKE_EQUIV:
		if (task_active_pid_ns(current) != &init_pid_ns)
			return -EPERM;

		if (!netlink_capable(skb, CAP_AUDIT_CONTROL))
			err = -EPERM;
		break;
	case AUDIT_USER:
	case AUDIT_FIRST_USER_MSG ... AUDIT_LAST_USER_MSG:
	case AUDIT_FIRST_USER_MSG2 ... AUDIT_LAST_USER_MSG2:
		if (!netlink_capable(skb, CAP_AUDIT_WRITE))
			err = -EPERM;
		break;
	default: 
		err = -EINVAL;
	}

	return err;
}

static void audit_log_common_recv_msg(struct audit_buffer **ab, u16 msg_type)
{
	uid_t uid = from_kuid(&init_user_ns, current_uid());
	pid_t pid = task_tgid_nr(current);

	if (!audit_enabled && msg_type != AUDIT_USER_AVC) {
		*ab = NULL;
		return;
	}

	*ab = audit_log_start(NULL, GFP_KERNEL, msg_type);
	if (unlikely(!*ab))
		return;
	audit_log_format(*ab, "pid=%d uid=%u", pid, uid);
	audit_log_session_info(*ab);
	audit_log_task_context(*ab);
}

int is_audit_feature_set(int i)
{
	return af.features & AUDIT_FEATURE_TO_MASK(i);
}


static int audit_get_feature(struct sk_buff *skb)
{
	u32 seq;

	seq = nlmsg_hdr(skb)->nlmsg_seq;

	audit_send_reply(skb, seq, AUDIT_GET_FEATURE, 0, 0, &af, sizeof(af));

	return 0;
}

static void audit_log_feature_change(int which, u32 old_feature, u32 new_feature,
				     u32 old_lock, u32 new_lock, int res)
{
	struct audit_buffer *ab;

	if (audit_enabled == AUDIT_OFF)
		return;

	ab = audit_log_start(NULL, GFP_KERNEL, AUDIT_FEATURE_CHANGE);
	audit_log_task_info(ab, current);
	audit_log_format(ab, " feature=%s old=%u new=%u old_lock=%u new_lock=%u res=%d",
			 audit_feature_names[which], !!old_feature, !!new_feature,
			 !!old_lock, !!new_lock, res);
	audit_log_end(ab);
}

static int audit_set_feature(struct sk_buff *skb)
{
	struct audit_features *uaf;
	int i;

	BUILD_BUG_ON(AUDIT_LAST_FEATURE + 1 > ARRAY_SIZE(audit_feature_names));
	uaf = nlmsg_data(nlmsg_hdr(skb));


	for (i = 0; i <= AUDIT_LAST_FEATURE; i++) {
		u32 feature = AUDIT_FEATURE_TO_MASK(i);
		u32 old_feature, new_feature, old_lock, new_lock;

		if (!(feature & uaf->mask))
			continue;

		old_feature = af.features & feature;
		new_feature = uaf->features & feature;
		new_lock = (uaf->lock | af.lock) & feature;
		old_lock = af.lock & feature;

		if (old_lock && (new_feature != old_feature)) {
			audit_log_feature_change(i, old_feature, new_feature,
						 old_lock, new_lock, 0);
			return -EPERM;
		}
	}
	for (i = 0; i <= AUDIT_LAST_FEATURE; i++) {
		u32 feature = AUDIT_FEATURE_TO_MASK(i);
		u32 old_feature, new_feature, old_lock, new_lock;

		if (!(feature & uaf->mask))
			continue;

		old_feature = af.features & feature;
		new_feature = uaf->features & feature;
		old_lock = af.lock & feature;
		new_lock = (uaf->lock | af.lock) & feature;

		if (new_feature != old_feature)
			audit_log_feature_change(i, old_feature, new_feature,
						 old_lock, new_lock, 1);

		if (new_feature)
			af.features |= feature;
		else
			af.features &= ~feature;
		af.lock |= new_lock;
	}

	return 0;
}

static int audit_receive_msg(struct sk_buff *skb, struct nlmsghdr *nlh)
{
	u32			seq;
	void			*data;
	int			err;
	struct audit_buffer	*ab;
	u16			msg_type = nlh->nlmsg_type;
	struct audit_sig_info   *sig_data;
	char			*ctx = NULL;
	u32			len;

	err = audit_netlink_ok(skb, msg_type);
	if (err)
		return err;

	if (!kauditd_task) {
		kauditd_task = kthread_run(kauditd_thread, NULL, "kauditd");
		if (IS_ERR(kauditd_task)) {
			err = PTR_ERR(kauditd_task);
			kauditd_task = NULL;
			return err;
		}
	}
	seq  = nlh->nlmsg_seq;
	data = nlmsg_data(nlh);

	switch (msg_type) {
	case AUDIT_GET: {
		struct audit_status	s;
		memset(&s, 0, sizeof(s));
		s.enabled		= audit_enabled;
		s.failure		= audit_failure;
		s.pid			= audit_pid;
		s.rate_limit		= audit_rate_limit;
		s.backlog_limit		= audit_backlog_limit;
		s.lost			= atomic_read(&audit_lost);
		s.backlog		= skb_queue_len(&audit_skb_queue);
		s.feature_bitmap	= AUDIT_FEATURE_BITMAP_ALL;
		s.backlog_wait_time	= audit_backlog_wait_time_master;
		audit_send_reply(skb, seq, AUDIT_GET, 0, 0, &s, sizeof(s));
		break;
	}
	case AUDIT_SET: {
		struct audit_status	s;
		memset(&s, 0, sizeof(s));
		memcpy(&s, data, min_t(size_t, sizeof(s), nlmsg_len(nlh)));
		if (s.mask & AUDIT_STATUS_ENABLED) {
			err = audit_set_enabled(s.enabled);
			if (err < 0)
				return err;
		}
		if (s.mask & AUDIT_STATUS_FAILURE) {
			err = audit_set_failure(s.failure);
			if (err < 0)
				return err;
		}
		if (s.mask & AUDIT_STATUS_PID) {
			int new_pid = s.pid;

			if ((!new_pid) && (task_tgid_vnr(current) != audit_pid))
				return -EACCES;
			if (audit_enabled != AUDIT_OFF)
				audit_log_config_change("audit_pid", new_pid, audit_pid, 1);
			audit_pid = new_pid;
			audit_nlk_portid = NETLINK_CB(skb).portid;
			audit_sock = skb->sk;
		}
		if (s.mask & AUDIT_STATUS_RATE_LIMIT) {
			err = audit_set_rate_limit(s.rate_limit);
			if (err < 0)
				return err;
		}
		if (s.mask & AUDIT_STATUS_BACKLOG_LIMIT) {
			err = audit_set_backlog_limit(s.backlog_limit);
			if (err < 0)
				return err;
		}
		if (s.mask & AUDIT_STATUS_BACKLOG_WAIT_TIME) {
			if (sizeof(s) > (size_t)nlh->nlmsg_len)
				return -EINVAL;
			if (s.backlog_wait_time > 10*AUDIT_BACKLOG_WAIT_TIME)
				return -EINVAL;
			err = audit_set_backlog_wait_time(s.backlog_wait_time);
			if (err < 0)
				return err;
		}
		break;
	}
	case AUDIT_GET_FEATURE:
		err = audit_get_feature(skb);
		if (err)
			return err;
		break;
	case AUDIT_SET_FEATURE:
		err = audit_set_feature(skb);
		if (err)
			return err;
		break;
	case AUDIT_USER:
	case AUDIT_FIRST_USER_MSG ... AUDIT_LAST_USER_MSG:
	case AUDIT_FIRST_USER_MSG2 ... AUDIT_LAST_USER_MSG2:
		if (!audit_enabled && msg_type != AUDIT_USER_AVC)
			return 0;

		err = audit_filter_user(msg_type);
		if (err == 1) {
			err = 0;
			if (msg_type == AUDIT_USER_TTY) {
				err = tty_audit_push_current();
				if (err)
					break;
			}
			mutex_unlock(&audit_cmd_mutex);
			audit_log_common_recv_msg(&ab, msg_type);
			if (msg_type != AUDIT_USER_TTY)
				audit_log_format(ab, " msg='%.*s'",
						 AUDIT_MESSAGE_TEXT_MAX,
						 (char *)data);
			else {
				int size;

				audit_log_format(ab, " data=");
				size = nlmsg_len(nlh);
				if (size > 0 &&
				    ((unsigned char *)data)[size - 1] == '\0')
					size--;
				audit_log_n_untrustedstring(ab, data, size);
			}
			audit_set_portid(ab, NETLINK_CB(skb).portid);
			audit_log_end(ab);
			mutex_lock(&audit_cmd_mutex);
		}
		break;
	case AUDIT_ADD_RULE:
	case AUDIT_DEL_RULE:
		if (nlmsg_len(nlh) < sizeof(struct audit_rule_data))
			return -EINVAL;
		if (audit_enabled == AUDIT_LOCKED) {
			audit_log_common_recv_msg(&ab, AUDIT_CONFIG_CHANGE);
			audit_log_format(ab, " audit_enabled=%d res=0", audit_enabled);
			audit_log_end(ab);
			return -EPERM;
		}
		err = audit_rule_change(msg_type, NETLINK_CB(skb).portid,
					   seq, data, nlmsg_len(nlh));
		break;
	case AUDIT_LIST_RULES:
		err = audit_list_rules_send(skb, seq);
		break;
	case AUDIT_TRIM:
		audit_trim_trees();
		audit_log_common_recv_msg(&ab, AUDIT_CONFIG_CHANGE);
		audit_log_format(ab, " op=trim res=1");
		audit_log_end(ab);
		break;
	case AUDIT_MAKE_EQUIV: {
		void *bufp = data;
		u32 sizes[2];
		size_t msglen = nlmsg_len(nlh);
		char *old, *new;

		err = -EINVAL;
		if (msglen < 2 * sizeof(u32))
			break;
		memcpy(sizes, bufp, 2 * sizeof(u32));
		bufp += 2 * sizeof(u32);
		msglen -= 2 * sizeof(u32);
		old = audit_unpack_string(&bufp, &msglen, sizes[0]);
		if (IS_ERR(old)) {
			err = PTR_ERR(old);
			break;
		}
		new = audit_unpack_string(&bufp, &msglen, sizes[1]);
		if (IS_ERR(new)) {
			err = PTR_ERR(new);
			kfree(old);
			break;
		}
		err = audit_tag_tree(old, new);

		audit_log_common_recv_msg(&ab, AUDIT_CONFIG_CHANGE);

		audit_log_format(ab, " op=make_equiv old=");
		audit_log_untrustedstring(ab, old);
		audit_log_format(ab, " new=");
		audit_log_untrustedstring(ab, new);
		audit_log_format(ab, " res=%d", !err);
		audit_log_end(ab);
		kfree(old);
		kfree(new);
		break;
	}
	case AUDIT_SIGNAL_INFO:
		len = 0;
		if (audit_sig_sid) {
			err = security_secid_to_secctx(audit_sig_sid, &ctx, &len);
			if (err)
				return err;
		}
		sig_data = kmalloc(sizeof(*sig_data) + len, GFP_KERNEL);
		if (!sig_data) {
			if (audit_sig_sid)
				security_release_secctx(ctx, len);
			return -ENOMEM;
		}
		sig_data->uid = from_kuid(&init_user_ns, audit_sig_uid);
		sig_data->pid = audit_sig_pid;
		if (audit_sig_sid) {
			memcpy(sig_data->ctx, ctx, len);
			security_release_secctx(ctx, len);
		}
		audit_send_reply(skb, seq, AUDIT_SIGNAL_INFO, 0, 0,
				 sig_data, sizeof(*sig_data) + len);
		kfree(sig_data);
		break;
	case AUDIT_TTY_GET: {
		struct audit_tty_status s;
		struct task_struct *tsk = current;

		spin_lock(&tsk->sighand->siglock);
		s.enabled = tsk->signal->audit_tty;
		s.log_passwd = tsk->signal->audit_tty_log_passwd;
		spin_unlock(&tsk->sighand->siglock);

		audit_send_reply(skb, seq, AUDIT_TTY_GET, 0, 0, &s, sizeof(s));
		break;
	}
	case AUDIT_TTY_SET: {
		struct audit_tty_status s, old;
		struct task_struct *tsk = current;
		struct audit_buffer	*ab;

		memset(&s, 0, sizeof(s));
		memcpy(&s, data, min_t(size_t, sizeof(s), nlmsg_len(nlh)));
		if ((s.enabled != 0 && s.enabled != 1) ||
		    (s.log_passwd != 0 && s.log_passwd != 1))
			err = -EINVAL;

		spin_lock(&tsk->sighand->siglock);
		old.enabled = tsk->signal->audit_tty;
		old.log_passwd = tsk->signal->audit_tty_log_passwd;
		if (!err) {
			tsk->signal->audit_tty = s.enabled;
			tsk->signal->audit_tty_log_passwd = s.log_passwd;
		}
		spin_unlock(&tsk->sighand->siglock);

		audit_log_common_recv_msg(&ab, AUDIT_CONFIG_CHANGE);
		audit_log_format(ab, " op=tty_set old-enabled=%d new-enabled=%d"
				 " old-log_passwd=%d new-log_passwd=%d res=%d",
				 old.enabled, s.enabled, old.log_passwd,
				 s.log_passwd, !err);
		audit_log_end(ab);
		break;
	}
	default:
		err = -EINVAL;
		break;
	}

	return err < 0 ? err : 0;
}

static void audit_receive_skb(struct sk_buff *skb)
{
	struct nlmsghdr *nlh;
	int len;
	int err;

	nlh = nlmsg_hdr(skb);
	len = skb->len;

	while (nlmsg_ok(nlh, len)) {
		err = audit_receive_msg(skb, nlh);
		if (err || (nlh->nlmsg_flags & NLM_F_ACK))
			netlink_ack(skb, nlh, err);

		nlh = nlmsg_next(nlh, &len);
	}
}

static void audit_receive(struct sk_buff  *skb)
{
	mutex_lock(&audit_cmd_mutex);
	audit_receive_skb(skb);
	mutex_unlock(&audit_cmd_mutex);
}

static int audit_bind(struct net *net, int group)
{
	if (!capable(CAP_AUDIT_READ))
		return -EPERM;

	return 0;
}

static int __net_init audit_net_init(struct net *net)
{
	struct netlink_kernel_cfg cfg = {
		.input	= audit_receive,
		.bind	= audit_bind,
		.flags	= NL_CFG_F_NONROOT_RECV,
		.groups	= AUDIT_NLGRP_MAX,
	};

	struct audit_net *aunet = net_generic(net, audit_net_id);

	aunet->nlsk = netlink_kernel_create(net, NETLINK_AUDIT, &cfg);
	if (aunet->nlsk == NULL) {
		audit_panic("cannot initialize netlink socket in namespace");
		return -ENOMEM;
	}
	aunet->nlsk->sk_sndtimeo = MAX_SCHEDULE_TIMEOUT;
	return 0;
}

static void __net_exit audit_net_exit(struct net *net)
{
	struct audit_net *aunet = net_generic(net, audit_net_id);
	struct sock *sock = aunet->nlsk;
	if (sock == audit_sock) {
		audit_pid = 0;
		audit_sock = NULL;
	}

	RCU_INIT_POINTER(aunet->nlsk, NULL);
	synchronize_net();
	netlink_kernel_release(sock);
}

static struct pernet_operations audit_net_ops __net_initdata = {
	.init = audit_net_init,
	.exit = audit_net_exit,
	.id = &audit_net_id,
	.size = sizeof(struct audit_net),
};

static int __init audit_init(void)
{
	int i;

	if (audit_initialized == AUDIT_DISABLED)
		return 0;

	pr_info("initializing netlink subsys (%s)\n",
		audit_default ? "enabled" : "disabled");
	register_pernet_subsys(&audit_net_ops);

	skb_queue_head_init(&audit_skb_queue);
	skb_queue_head_init(&audit_skb_hold_queue);
	audit_initialized = AUDIT_INITIALIZED;
	audit_enabled = audit_default;
	audit_ever_enabled |= !!audit_default;

	audit_log(NULL, GFP_KERNEL, AUDIT_KERNEL, "initialized");

	for (i = 0; i < AUDIT_INODE_BUCKETS; i++)
		INIT_LIST_HEAD(&audit_inode_hash[i]);

	return 0;
}
__initcall(audit_init);

static int __init audit_enable(char *str)
{
	audit_default = !!simple_strtol(str, NULL, 0);
	if (!audit_default)
		audit_initialized = AUDIT_DISABLED;

	pr_info("%s\n", audit_default ?
		"enabled (after initialization)" : "disabled (until reboot)");

	return 1;
}
__setup("audit=", audit_enable);

static int __init audit_backlog_limit_set(char *str)
{
	u32 audit_backlog_limit_arg;

	pr_info("audit_backlog_limit: ");
	if (kstrtouint(str, 0, &audit_backlog_limit_arg)) {
		pr_cont("using default of %u, unable to parse %s\n",
			audit_backlog_limit, str);
		return 1;
	}

	audit_backlog_limit = audit_backlog_limit_arg;
	pr_cont("%d\n", audit_backlog_limit);

	return 1;
}
__setup("audit_backlog_limit=", audit_backlog_limit_set);

static void audit_buffer_free(struct audit_buffer *ab)
{
	unsigned long flags;

	if (!ab)
		return;

	kfree_skb(ab->skb);
	spin_lock_irqsave(&audit_freelist_lock, flags);
	if (audit_freelist_count > AUDIT_MAXFREE)
		kfree(ab);
	else {
		audit_freelist_count++;
		list_add(&ab->list, &audit_freelist);
	}
	spin_unlock_irqrestore(&audit_freelist_lock, flags);
}

static struct audit_buffer * audit_buffer_alloc(struct audit_context *ctx,
						gfp_t gfp_mask, int type)
{
	unsigned long flags;
	struct audit_buffer *ab = NULL;
	struct nlmsghdr *nlh;

	spin_lock_irqsave(&audit_freelist_lock, flags);
	if (!list_empty(&audit_freelist)) {
		ab = list_entry(audit_freelist.next,
				struct audit_buffer, list);
		list_del(&ab->list);
		--audit_freelist_count;
	}
	spin_unlock_irqrestore(&audit_freelist_lock, flags);

	if (!ab) {
		ab = kmalloc(sizeof(*ab), gfp_mask);
		if (!ab)
			goto err;
	}

	ab->ctx = ctx;
	ab->gfp_mask = gfp_mask;

	ab->skb = nlmsg_new(AUDIT_BUFSIZ, gfp_mask);
	if (!ab->skb)
		goto err;

	nlh = nlmsg_put(ab->skb, 0, 0, type, 0, 0);
	if (!nlh)
		goto out_kfree_skb;

	return ab;

out_kfree_skb:
	kfree_skb(ab->skb);
	ab->skb = NULL;
err:
	audit_buffer_free(ab);
	return NULL;
}

unsigned int audit_serial(void)
{
	static atomic_t serial = ATOMIC_INIT(0);

	return atomic_add_return(1, &serial);
}

static inline void audit_get_stamp(struct audit_context *ctx,
				   struct timespec *t, unsigned int *serial)
{
	if (!ctx || !auditsc_get_stamp(ctx, t, serial)) {
		*t = CURRENT_TIME;
		*serial = audit_serial();
	}
}

static long wait_for_auditd(long sleep_time)
{
	DECLARE_WAITQUEUE(wait, current);
	set_current_state(TASK_UNINTERRUPTIBLE);
	add_wait_queue_exclusive(&audit_backlog_wait, &wait);

	if (audit_backlog_limit &&
	    skb_queue_len(&audit_skb_queue) > audit_backlog_limit)
		sleep_time = schedule_timeout(sleep_time);

	__set_current_state(TASK_RUNNING);
	remove_wait_queue(&audit_backlog_wait, &wait);

	return sleep_time;
}

struct audit_buffer *audit_log_start(struct audit_context *ctx, gfp_t gfp_mask,
				     int type)
{
	struct audit_buffer	*ab	= NULL;
	struct timespec		t;
	unsigned int		uninitialized_var(serial);
	int reserve = 5; 
			    
	unsigned long timeout_start = jiffies;

	if (audit_initialized != AUDIT_INITIALIZED)
		return NULL;

	if (unlikely(audit_filter_type(type)))
		return NULL;

	if (gfp_mask & __GFP_DIRECT_RECLAIM) {
		if (audit_pid && audit_pid == current->tgid)
			gfp_mask &= ~__GFP_DIRECT_RECLAIM;
		else
			reserve = 0;
	}

	while (audit_backlog_limit
	       && skb_queue_len(&audit_skb_queue) > audit_backlog_limit + reserve) {
		if (gfp_mask & __GFP_DIRECT_RECLAIM && audit_backlog_wait_time) {
			long sleep_time;

			sleep_time = timeout_start + audit_backlog_wait_time - jiffies;
			if (sleep_time > 0) {
				sleep_time = wait_for_auditd(sleep_time);
				if (sleep_time > 0)
					continue;
			}
		}
		if (audit_rate_check() && printk_ratelimit())
			pr_warn("audit_backlog=%d > audit_backlog_limit=%d\n",
				skb_queue_len(&audit_skb_queue),
				audit_backlog_limit);
		audit_log_lost("backlog limit exceeded");
		audit_backlog_wait_time = 0;
		wake_up(&audit_backlog_wait);
		return NULL;
	}

	if (!reserve && !audit_backlog_wait_time)
		audit_backlog_wait_time = audit_backlog_wait_time_master;

	ab = audit_buffer_alloc(ctx, gfp_mask, type);
	if (!ab) {
		audit_log_lost("out of memory in audit_log_start");
		return NULL;
	}

	audit_get_stamp(ab->ctx, &t, &serial);

	audit_log_format(ab, "audit(%lu.%03lu:%u): ",
			 t.tv_sec, t.tv_nsec/1000000, serial);
	return ab;
}

static inline int audit_expand(struct audit_buffer *ab, int extra)
{
	struct sk_buff *skb = ab->skb;
	int oldtail = skb_tailroom(skb);
	int ret = pskb_expand_head(skb, 0, extra, ab->gfp_mask);
	int newtail = skb_tailroom(skb);

	if (ret < 0) {
		audit_log_lost("out of memory in audit_expand");
		return 0;
	}

	skb->truesize += newtail - oldtail;
	return newtail;
}

static void audit_log_vformat(struct audit_buffer *ab, const char *fmt,
			      va_list args)
{
	int len, avail;
	struct sk_buff *skb;
	va_list args2;

	if (!ab)
		return;

	BUG_ON(!ab->skb);
	skb = ab->skb;
	avail = skb_tailroom(skb);
	if (avail == 0) {
		avail = audit_expand(ab, AUDIT_BUFSIZ);
		if (!avail)
			goto out;
	}
	va_copy(args2, args);
	len = vsnprintf(skb_tail_pointer(skb), avail, fmt, args);
	if (len >= avail) {
		avail = audit_expand(ab,
			max_t(unsigned, AUDIT_BUFSIZ, 1+len-avail));
		if (!avail)
			goto out_va_end;
		len = vsnprintf(skb_tail_pointer(skb), avail, fmt, args2);
	}
	if (len > 0)
		skb_put(skb, len);
out_va_end:
	va_end(args2);
out:
	return;
}

void audit_log_format(struct audit_buffer *ab, const char *fmt, ...)
{
	va_list args;

	if (!ab)
		return;
	va_start(args, fmt);
	audit_log_vformat(ab, fmt, args);
	va_end(args);
}

void audit_log_n_hex(struct audit_buffer *ab, const unsigned char *buf,
		size_t len)
{
	int i, avail, new_len;
	unsigned char *ptr;
	struct sk_buff *skb;

	if (!ab)
		return;

	BUG_ON(!ab->skb);
	skb = ab->skb;
	avail = skb_tailroom(skb);
	new_len = len<<1;
	if (new_len >= avail) {
		new_len = AUDIT_BUFSIZ*(((new_len-avail)/AUDIT_BUFSIZ) + 1);
		avail = audit_expand(ab, new_len);
		if (!avail)
			return;
	}

	ptr = skb_tail_pointer(skb);
	for (i = 0; i < len; i++)
		ptr = hex_byte_pack_upper(ptr, buf[i]);
	*ptr = 0;
	skb_put(skb, len << 1); 
}

void audit_log_n_string(struct audit_buffer *ab, const char *string,
			size_t slen)
{
	int avail, new_len;
	unsigned char *ptr;
	struct sk_buff *skb;

	if (!ab)
		return;

	BUG_ON(!ab->skb);
	skb = ab->skb;
	avail = skb_tailroom(skb);
	new_len = slen + 3;	
	if (new_len > avail) {
		avail = audit_expand(ab, new_len);
		if (!avail)
			return;
	}
	ptr = skb_tail_pointer(skb);
	*ptr++ = '"';
	memcpy(ptr, string, slen);
	ptr += slen;
	*ptr++ = '"';
	*ptr = 0;
	skb_put(skb, slen + 2);	
}

bool audit_string_contains_control(const char *string, size_t len)
{
	const unsigned char *p;
	for (p = string; p < (const unsigned char *)string + len; p++) {
		if (*p == '"' || *p < 0x21 || *p > 0x7e)
			return true;
	}
	return false;
}

void audit_log_n_untrustedstring(struct audit_buffer *ab, const char *string,
				 size_t len)
{
	if (audit_string_contains_control(string, len))
		audit_log_n_hex(ab, string, len);
	else
		audit_log_n_string(ab, string, len);
}

void audit_log_untrustedstring(struct audit_buffer *ab, const char *string)
{
	audit_log_n_untrustedstring(ab, string, strlen(string));
}

void audit_log_d_path(struct audit_buffer *ab, const char *prefix,
		      const struct path *path)
{
	char *p, *pathname;

	if (prefix)
		audit_log_format(ab, "%s", prefix);

	pathname = kmalloc(PATH_MAX+11, ab->gfp_mask);
	if (!pathname) {
		audit_log_string(ab, "<no_memory>");
		return;
	}
	p = d_path(path, pathname, PATH_MAX+11);
	if (IS_ERR(p)) {
		audit_log_string(ab, "<too_long>");
	} else
		audit_log_untrustedstring(ab, p);
	kfree(pathname);
}

void audit_log_session_info(struct audit_buffer *ab)
{
	unsigned int sessionid = audit_get_sessionid(current);
	uid_t auid = from_kuid(&init_user_ns, audit_get_loginuid(current));

	audit_log_format(ab, " auid=%u ses=%u", auid, sessionid);
}

void audit_log_key(struct audit_buffer *ab, char *key)
{
	audit_log_format(ab, " key=");
	if (key)
		audit_log_untrustedstring(ab, key);
	else
		audit_log_format(ab, "(null)");
}

void audit_log_cap(struct audit_buffer *ab, char *prefix, kernel_cap_t *cap)
{
	int i;

	audit_log_format(ab, " %s=", prefix);
	CAP_FOR_EACH_U32(i) {
		audit_log_format(ab, "%08x",
				 cap->cap[CAP_LAST_U32 - i]);
	}
}

static void audit_log_fcaps(struct audit_buffer *ab, struct audit_names *name)
{
	kernel_cap_t *perm = &name->fcap.permitted;
	kernel_cap_t *inh = &name->fcap.inheritable;
	int log = 0;

	if (!cap_isclear(*perm)) {
		audit_log_cap(ab, "cap_fp", perm);
		log = 1;
	}
	if (!cap_isclear(*inh)) {
		audit_log_cap(ab, "cap_fi", inh);
		log = 1;
	}

	if (log)
		audit_log_format(ab, " cap_fe=%d cap_fver=%x",
				 name->fcap.fE, name->fcap_ver);
}

static inline int audit_copy_fcaps(struct audit_names *name,
				   const struct dentry *dentry)
{
	struct cpu_vfs_cap_data caps;
	int rc;

	if (!dentry)
		return 0;

	rc = get_vfs_caps_from_disk(dentry, &caps);
	if (rc)
		return rc;

	name->fcap.permitted = caps.permitted;
	name->fcap.inheritable = caps.inheritable;
	name->fcap.fE = !!(caps.magic_etc & VFS_CAP_FLAGS_EFFECTIVE);
	name->fcap_ver = (caps.magic_etc & VFS_CAP_REVISION_MASK) >>
				VFS_CAP_REVISION_SHIFT;

	return 0;
}

void audit_copy_inode(struct audit_names *name, const struct dentry *dentry,
		      struct inode *inode)
{
	name->ino   = inode->i_ino;
	name->dev   = inode->i_sb->s_dev;
	name->mode  = inode->i_mode;
	name->uid   = inode->i_uid;
	name->gid   = inode->i_gid;
	name->rdev  = inode->i_rdev;
	security_inode_getsecid(inode, &name->osid);
	audit_copy_fcaps(name, dentry);
}


void audit_log_name(struct audit_context *context, struct audit_names *n,
		    struct path *path, int record_num, int *call_panic)
{
	struct audit_buffer *ab;
	ab = audit_log_start(context, GFP_KERNEL, AUDIT_PATH);
	if (!ab)
		return;

	audit_log_format(ab, "item=%d", record_num);

	if (path)
		audit_log_d_path(ab, " name=", path);
	else if (n->name) {
		switch (n->name_len) {
		case AUDIT_NAME_FULL:
			audit_log_format(ab, " name=");
			audit_log_untrustedstring(ab, n->name->name);
			break;
		case 0:
			audit_log_d_path(ab, " name=", &context->pwd);
			break;
		default:
			audit_log_format(ab, " name=");
			audit_log_n_untrustedstring(ab, n->name->name,
						    n->name_len);
		}
	} else
		audit_log_format(ab, " name=(null)");

	if (n->ino != AUDIT_INO_UNSET)
		audit_log_format(ab, " inode=%lu"
				 " dev=%02x:%02x mode=%#ho"
				 " ouid=%u ogid=%u rdev=%02x:%02x",
				 n->ino,
				 MAJOR(n->dev),
				 MINOR(n->dev),
				 n->mode,
				 from_kuid(&init_user_ns, n->uid),
				 from_kgid(&init_user_ns, n->gid),
				 MAJOR(n->rdev),
				 MINOR(n->rdev));
	if (n->osid != 0) {
		char *ctx = NULL;
		u32 len;
		if (security_secid_to_secctx(
			n->osid, &ctx, &len)) {
			audit_log_format(ab, " osid=%u", n->osid);
			if (call_panic)
				*call_panic = 2;
		} else {
			audit_log_format(ab, " obj=%s", ctx);
			security_release_secctx(ctx, len);
		}
	}

	audit_log_format(ab, " nametype=");
	switch(n->type) {
	case AUDIT_TYPE_NORMAL:
		audit_log_format(ab, "NORMAL");
		break;
	case AUDIT_TYPE_PARENT:
		audit_log_format(ab, "PARENT");
		break;
	case AUDIT_TYPE_CHILD_DELETE:
		audit_log_format(ab, "DELETE");
		break;
	case AUDIT_TYPE_CHILD_CREATE:
		audit_log_format(ab, "CREATE");
		break;
	default:
		audit_log_format(ab, "UNKNOWN");
		break;
	}

	audit_log_fcaps(ab, n);
	audit_log_end(ab);
}

int audit_log_task_context(struct audit_buffer *ab)
{
	char *ctx = NULL;
	unsigned len;
	int error;
	u32 sid;

	security_task_getsecid(current, &sid);
	if (!sid)
		return 0;

	error = security_secid_to_secctx(sid, &ctx, &len);
	if (error) {
		if (error != -EINVAL)
			goto error_path;
		return 0;
	}

	audit_log_format(ab, " subj=%s", ctx);
	security_release_secctx(ctx, len);
	return 0;

error_path:
	audit_panic("error in audit_log_task_context");
	return error;
}
EXPORT_SYMBOL(audit_log_task_context);

void audit_log_d_path_exe(struct audit_buffer *ab,
			  struct mm_struct *mm)
{
	struct file *exe_file;

	if (!mm)
		goto out_null;

	exe_file = get_mm_exe_file(mm);
	if (!exe_file)
		goto out_null;

	audit_log_d_path(ab, " exe=", &exe_file->f_path);
	fput(exe_file);
	return;
out_null:
	audit_log_format(ab, " exe=(null)");
}

void audit_log_task_info(struct audit_buffer *ab, struct task_struct *tsk)
{
	const struct cred *cred;
	char comm[sizeof(tsk->comm)];
	char *tty;

	if (!ab)
		return;

	cred = current_cred();

	spin_lock_irq(&tsk->sighand->siglock);
	if (tsk->signal && tsk->signal->tty && tsk->signal->tty->name)
		tty = tsk->signal->tty->name;
	else
		tty = "(none)";
	spin_unlock_irq(&tsk->sighand->siglock);

	audit_log_format(ab,
			 " ppid=%d pid=%d auid=%u uid=%u gid=%u"
			 " euid=%u suid=%u fsuid=%u"
			 " egid=%u sgid=%u fsgid=%u tty=%s ses=%u",
			 task_ppid_nr(tsk),
			 task_pid_nr(tsk),
			 from_kuid(&init_user_ns, audit_get_loginuid(tsk)),
			 from_kuid(&init_user_ns, cred->uid),
			 from_kgid(&init_user_ns, cred->gid),
			 from_kuid(&init_user_ns, cred->euid),
			 from_kuid(&init_user_ns, cred->suid),
			 from_kuid(&init_user_ns, cred->fsuid),
			 from_kgid(&init_user_ns, cred->egid),
			 from_kgid(&init_user_ns, cred->sgid),
			 from_kgid(&init_user_ns, cred->fsgid),
			 tty, audit_get_sessionid(tsk));

	audit_log_format(ab, " comm=");
	audit_log_untrustedstring(ab, get_task_comm(comm, tsk));

	audit_log_d_path_exe(ab, tsk->mm);
	audit_log_task_context(ab);
}
EXPORT_SYMBOL(audit_log_task_info);


void audit_log_link_denied(const char *operation, struct path *link)
{
	struct audit_buffer *ab;
	struct audit_names *name;

	name = kzalloc(sizeof(*name), GFP_NOFS);
	if (!name)
		return;

	ab = audit_log_start(current->audit_context, GFP_KERNEL,
			     AUDIT_ANOM_LINK);
	if (!ab)
		goto out;
	audit_log_format(ab, "op=%s", operation);
	audit_log_task_info(ab, current);
	audit_log_format(ab, " res=0");
	audit_log_end(ab);

	name->type = AUDIT_TYPE_NORMAL;
	audit_copy_inode(name, link->dentry, d_backing_inode(link->dentry));
	audit_log_name(current->audit_context, name, link, 0, NULL);
out:
	kfree(name);
}

void audit_log_end(struct audit_buffer *ab)
{
	if (!ab)
		return;
	if (!audit_rate_check()) {
		audit_log_lost("rate limit exceeded");
	} else {
		struct nlmsghdr *nlh = nlmsg_hdr(ab->skb);

		nlh->nlmsg_len = ab->skb->len;
		kauditd_send_multicast_skb(ab->skb, ab->gfp_mask);

		
		nlh->nlmsg_len -= NLMSG_HDRLEN;

		if (audit_pid) {
			skb_queue_tail(&audit_skb_queue, ab->skb);
			wake_up_interruptible(&kauditd_wait);
		} else {
			audit_printk_skb(ab->skb);
		}
		ab->skb = NULL;
	}
	audit_buffer_free(ab);
}

void audit_log(struct audit_context *ctx, gfp_t gfp_mask, int type,
	       const char *fmt, ...)
{
	struct audit_buffer *ab;
	va_list args;

	ab = audit_log_start(ctx, gfp_mask, type);
	if (ab) {
		va_start(args, fmt);
		audit_log_vformat(ab, fmt, args);
		va_end(args);
		audit_log_end(ab);
	}
}

#ifdef CONFIG_SECURITY

void audit_log_secctx(struct audit_buffer *ab, u32 secid)
{
	u32 len;
	char *secctx;

	if (security_secid_to_secctx(secid, &secctx, &len)) {
		audit_panic("Cannot convert secid to context");
	} else {
		audit_log_format(ab, " obj=%s", secctx);
		security_release_secctx(secctx, len);
	}
}
EXPORT_SYMBOL(audit_log_secctx);
#endif

EXPORT_SYMBOL(audit_log_start);
EXPORT_SYMBOL(audit_log_end);
EXPORT_SYMBOL(audit_log_format);
EXPORT_SYMBOL(audit_log);