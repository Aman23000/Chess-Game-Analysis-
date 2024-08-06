# Chess Game Analysis Using PySpark and RDDs

## Project Overview
This project aims to analyze large datasets using big data technologies, including Resilient Distributed Datasets (RDDs) and PySpark. The objective is to gain insights into player strategy, game dynamics, and performance trends in chess games.

## Files
- **spark_rdd.py**: The main Python script for data processing and machine learning analysis using PySpark.
- **sark_rdd_visualization**: THis Python script is for Visualization include all the dataprocessing and ml analysis using Pyspark but mainly for Visualization.
- **Amanjain_ProjectReport.pdf**: The project report detailing the objectives, methodology, results, and conclusions.
- **Presentation_amanjain_cs777 copy.pptx**: The PowerPoint presentation summarizing the project.
- **Plots, Result Screenshot, Spark UI Screenshot/**: Directory containing plots and screenshots of results.

## Setup Instructions
### Prerequisites
- Python 3.6 or higher
- PySpark
- Google Cloud SDK (if accessing data from Google Cloud Storage)
- Required Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `gcsfs`, `scikit-learn`

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/BigData-Analysis-Using-PySpark.git
   cd BigData-Analysis-Using-PySpark
2. Install PySpark:
   ```sh
   pip install pyspark
3. pip install pandas numpy matplotlib seaborn gcsfs scikit-learn
   ``` sh
   pip install pandas numpy matplotlib seaborn gcsfs scikit-learn
   
### Running the Project
1. Navigate to the project directory.
2. Ensure the dataset is accessible in the specified path in the pr3.py script. If using Google Cloud Storage, ensure the path is correctly set and the necessary permissions are in place.
3. Run the spark_rdd.py script : 
   python spark_rdd.py
4. Additionaly for the Visualization I have converted the results into pandas Dataframe and than plot the diagrams. 

### Data Source
The dataset used in this project is stored in Google Cloud Platform (GCP). You can access the dataset from the following link: https://www.kaggle.com/datasets/joannpeeler/labeled-chess-positions-109m-csv-format/data

### Results 
The results of the analysis are documented in the project report and presented in the PowerPoint presentation. Below are some key visualizations from the project:
Plots and Screenshots
Plot 1: Distribution of Evaluation Scores
Plot 2: Ply vs Eval
Plot 3: Histogram of Ply Distribution
Plot 4: Scatter Plot Matrix of Features
Plot 5: Boxplot of Ply by Result
Plot 6: Violin Plot of Eval by Result

### Project Structure
BigData-Analysis-Using-PySpark/
│
├── Plot and Screenshots
│   ├── eval_distribution.png
│   ├── ply_vs_eval.png
│   ├── ply_distribution.png
│   ├── scatter_matrix.png
│   ├── boxplot_ply_by_result.png
│   └── violinplot_eval_by_result.png
│
├── pr3.py
├── Amanjain_ProjectReport.pdf
├── Presentation_amanjain_cs777 copy.pptx
└── README.md

### Acknowledgments
1. "Machine Learning: A Probabilistic Perspective" by Kevin P. Murphy
2. Apache Spark Documentation
3. Google Cloud Platform Documentation
4. Kaggle Datasets and Discussions
5. Stack Overflow Community


This README file includes the link to the dataset and provides a comprehensive overview of the project, including setup instructions, running the project, results, and acknowledgments. Let me know if you need any further adjustments or additions!


