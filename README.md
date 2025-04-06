# Text Analysis Tool (FA_all)

A web-based application for analyzing text files using n-gram distributions, fluctuation analysis, and statistical characteristics.

## Features

- **Interactive Web Interface**: Built with Dash and Bootstrap for a modern, responsive UI
- **Multiple File Processing**: Analyze and compare multiple text files in batch mode
- **Flexible Text Parsing**: Process text as words, symbols, or letters
- **Advanced Statistical Analysis**: Calculate fluctuation characteristics, word distribution metrics, and correlations
- **Visualization**: Generate interactive plots and graphs for data representation
- **Markov Chain Analysis**: Model text using Markov chains and visualize word relationships
- **Adjustable Parameters**: Customize analysis with boundary conditions, window sizes, and filtering options
- **Data Export**: Save analysis results to Excel files for further processing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd FA_all
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- numpy (≥1.20.0)
- numba (≥0.54.0)
- matplotlib (≥3.4.0)
- pandas (≥1.3.0)
- openpyxl (≥3.0.7)
- scipy (≥1.7.0)
- dash (≥2.0.0)
- dash-bootstrap-components (≥1.0.0)
- plotly (≥5.0.0)
- networkx (≥2.6.0)
- scikit-learn (≥1.0.0)

## Usage

1. Run the application:
```bash
python FA_all.py
```

2. Open your browser and navigate to http://127.0.0.1:8050/

3. Upload text files for analysis:
   - Use the drag-and-drop area or file browser to upload one or more text files
   - Select how to split text (words, symbols, or letters)
   - Configure n-gram size and other parameters

4. Analyze files individually:
   - Select a file from the dropdown menu
   - Click "Analyze" to generate statistics for the selected file
   - View results in the DataTable tab
   - Explore the Markov Chain visualization in the MarkovChain tab

5. Process files in batch:
   - Set Fmin1 (filter for the shortest file) and Fmin2 (filter for the longest file)
   - Click "Process All Files" to analyze all uploaded files
   - Review results in the batch processing table
   - Save batch results to Excel with the "Save Batch Results" button

## Parameters

- **Size of ngram**: Number of words/symbols to treat as a unit (default: 1)
- **Split by**: How to divide the text (word, symbol, letter)
- **Boundary Condition**: How to handle boundaries (no, periodic, ordinary)
- **Min Distance**: Minimum distance between elements (0 or 1)
- **Window Mode**: Overlapping or non-overlapping sliding windows
- **Window Parameters**: Min window, window shift, expansion, and max window values
- **Definition**: Static or dynamic analysis mode

## Batch Processing

The application can automatically process multiple files and calculate appropriate filter values based on file lengths:

1. Set Lmin (Fmin1) and Lmax (Fmin2) values for minimum and maximum filtering
2. The system will calculate appropriate F_min values for each file using linear interpolation
3. Results will include:
   - File information (name, length, vocabulary size)
   - Eight statistical parameters (R_avg, dR, Rw_avg, dRw, γ_avg, dγ, γw_avg, dγw)
   - Mean and standard deviation across all files

## Outputs

For each analysis, the application calculates:
- Frequency distribution of n-grams
- Fluctuation metrics
- Rank-frequency relations
- Correlation patterns
- Statistical parameters for text characterization

Results can be exported to Excel files for further analysis and comparison. 