# NFL-Combine-Spectral-Clustering

# README

## Project Overview
This project provides tools to analyze and cluster NFL draft combine data from 2020 to 2024. It includes functionality to preprocess player statistics, create clusters based on performance metrics, and group players by similarity. The repository is designed for football analysts, coaches, and data enthusiasts to better understand player characteristics.

## Directory Structure
```
root/
|
|-- README.md             # This file
|-- LICENSE               # MIT license for the project
|-- src/
    |-- clusterer.py      # Core clustering implementation
    |-- clusterexamples.py # Example and usage of clustering
    |-- data/
        |-- draft2020.csv # Combine data for 2020
        |-- draft2021.csv # Combine data for 2021
        |-- draft2022.csv # Combine data for 2022
        |-- draft2023.csv # Combine data for 2023
        |-- draft2024.csv # Combine data for 2024
```

## Requirements
- Python 3.8 or higher
- Required libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   ```
2. Navigate to the project directory:
   ```bash
   cd your-repo-name
   ```

## Usage

### Core Functionality
The `src/clusterer.py` file contains the main clustering logic. You can use it to preprocess and cluster NFL combine data.

#### Example Workflow
The `src/clusterexamples.py` file provides examples of how to use the clustering functionality. To run the examples:
```bash
python src/clusterexamples.py
```

### Data
NFL combine data from 2020 to 2024 is stored in the `src/data/` directory. The data includes player statistics such as height, weight, 40-yard dash time, and other combine metrics.

### Adding New Data
To add combine data for a given year, navigate to [the Pro Football Reference Draft Page](https://www.pro-football-reference.com/draft/), select 'Combine Results', choose the year you'd like, click 'Share and Export', click Get table as CSF (for Excel)", write a new CSV file with the listed text as its contents, place the CSV file in the `src/data/` directory, and ensure it follows the same format as the existing files. The clustering script will automatically process it when specified.

## Data Sources
This project uses publicly available NFL combine data from [Pro Football Reference](https://www.pro-football-reference.com/)

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for review.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

