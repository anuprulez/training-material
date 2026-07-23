---
layout: tutorial_hands_on

title: Cancer prediction from gene expression with TabPFN
questions:
- How can platelet gene-expression profiles be prepared for multiclass cancer prediction?
- How can TabPFN classify high-dimensional biological data after feature selection?
- How does TabPFN compare with a conventional Random Forest baseline?
objectives:
- Prepare GSE68086 metadata and gene-expression data in a sample-by-feature table
- Encode cancer labels and select informative genes without scaling the expression values
- Create a stratified training and test split
- Train TabPFN and Random Forest classifiers and compare their confusion matrices
time_estimation: 2H
key_points:
- TabPFN performs in-context prediction with a pretrained model instead of fitting a new neural network from scratch
- The feature table must have samples in rows, the same ordered features for training and testing, and a clearly identified label column
- Feature selection must be performed before fitting either classifier and should normally be learned only from training data
- A shared stratified split makes model comparisons reproducible and preserves class proportions
contributors:
- anuprulez
subtopic: machine-learning
---


Tumor-educated platelets contain RNA patterns associated with the presence and type of a tumor. The GSE68086 study contains platelet RNA-seq profiles from patients with several cancers ({% cite Best2015 %}). In this tutorial, we use five cancer classes—breast cancer, colorectal cancer (CRC), lung cancer, glioblastoma (GBM), and pancreatic cancer—and ask whether their expression profiles can distinguish the cancer type.

The main classifier is TabPFN, a foundation model pretrained on synthetic tabular learning problems. At prediction time it uses the supplied training table as context, avoiding task-specific neural-network training ({% cite Hollmann2025 %}). We compare it with a Random Forest trained on exactly the same split.

> <agenda-title></agenda-title>
>
> In this tutorial, we will cover:
>
> 1. TOC
> {:toc}
>
{: .agenda}

# Get the data

### Source https://arxiv.org/pdf/2603.22675

 GSE68086: Bulk RNA-seq data

#### Another TabPFN reference paper: https://openreview.net/pdf?id=3Phk0nC9hK

The workflow needs the GEO series metadata, the expression matrix, and a small Jupyter notebook that reshapes them. The notebook retains the five cancer types and transposes the matrix so each row is a sample and each gene is a feature.

> <hands-on-title>Upload the inputs</hands-on-title>
>
> 1. Create a new history for this tutorial and give it a meaningful name.
>
>    {% snippet faqs/galaxy/histories_create_new.md %}
>
> 2. Import the following three files, or obtain them from the shared data library under `GTN - Material` → `Statistics and machine learning` → `Cancer prediction from gene expression with TabPFN`:
>
>    ```text
>    https://ftp.ncbi.nlm.nih.gov/geo/series/GSE68nnn/GSE68086/matrix/GSE68086_series_matrix.txt.gz
>    https://ftp.ncbi.nlm.nih.gov/geo/series/GSE68nnn/GSE68086/suppl/GSE68086_TEP_data_matrix.txt.gz
>    https://raw.githubusercontent.com/galaxyproject/training-material/main/topics/statistics/tutorials/tabpfn-gene-expression/preprocess_expression_data.ipynb
>    ```
>
>    {% snippet faqs/galaxy/datasets_import_via_link.md %}
>
>    {% snippet faqs/galaxy/datasets_import_from_data_library.md %}
>
> 3. Rename the files to `GSE68086 series file`, `GSE68086 matrix file`, and `preprocess_expression_data.ipynb`, respectively.
> 4. Set the notebook datatype to `ipynb`. Set both GEO datasets to `txt` if Galaxy did not detect a suitable text datatype.
>
{: .hands_on}

# Prepare the expression table

The series file stores sample annotations in rows beginning with `!Sample_characteristics_ch1`. The supplied notebook matches these annotations to the matrix columns, keeps five tumor types, and creates a table whose first column is `label`.

> <hands-on-title>Run the preprocessing notebook</hands-on-title>
>
> 1. {% tool [Interactive JupyterLab Notebook](interactive_tool_jupyter_notebook) %} with the following parameters:
>    - *"Create a new notebook or upload one?"*: `Load a previous Notebook`
>    - {% icon param-file %} *"IPython Notebook"*: `preprocess_expression_data.ipynb`
>    - In *"User inputs"*:
>      - Add an input named `series` and select `GSE68086 series file`
>      - Add another input named `gene_expression` and select `GSE68086 matrix file`
>    - *"Execute notebook?"*: `Yes`
> 2. Wait for the interactive job to finish. Rename its tabular output to `GSE68086 expression by sample`.
> 3. Inspect the table. It should start with `label`; the remaining columns are gene-expression features.
>
{: .hands_on}

> <question-title></question-title>
>
> 1. Why does the notebook transpose the original expression matrix?
> 2. Why are the sample accessions replaced with cancer types?
>
> > <solution-title></solution-title>
> >
> > 1. Galaxy's machine-learning tools expect observations (samples) in rows and variables (genes) in columns. The GEO matrix has the opposite orientation.
> > 2. The classifier needs a target for every sample. The cancer type becomes that target; the accession itself is only an identifier and must not be used as a predictive feature.
> >
> {: .solution}
>
{: .question}

# Encode labels and assemble the modeling table

We next separate the label and expression columns, encode the text labels as integers, and then join them again. This makes the target compatible with the machine-learning tools.

> <hands-on-title>Encode the cancer labels</hands-on-title>
>
> 1. {% tool [Cut](Cut1) %} from `GSE68086 expression by sample`:
>    - *"Cut columns"*: `c1`
> 2. Rename the output to `cancer labels`.
> 3. Run {% tool [Cut](Cut1) %} again on `GSE68086 expression by sample`:
>    - *"Cut columns"*: `c2-c57737`
> 4. Rename the output to `expression features`.
> 5. {% tool [Label encoder](toolshed.g2.bx.psu.edu/repos/bgruening/sklearn_label_encoder/sklearn_label_encoder/1.0.11.2) %}:
>    - {% icon param-file %} *"Select a file containing tabular data"*: `cancer labels`
>    - *"Does the dataset contain a header?"*: `Yes`
> 6. {% tool [Add line to file](toolshed.g2.bx.psu.edu/repos/bgruening/add_line_to_file/add_line_to_file/0.1.0) %}:
>    - {% icon param-file %} *"Input file"*: the encoded-label output
>    - *"Add line to"*: `Beginning of file`
>    - *"Text to add"*: `label`
> 7. Rename the result to `encoded labels`.
> 8. {% tool [Paste](Paste1) %}:
>    - {% icon param-file %} *"Paste file"*: `expression features`
>    - {% icon param-file %} *"and file"*: `encoded labels`
>    - *"Delimiter"*: `Tab`
> 9. Rename the result to `expression and labels`.
>
{: .hands_on}

# Select informative genes

The expression table contains tens of thousands of genes but only a few hundred samples. We use a chi-squared score to retain ten features. No scaling or other expression preprocessing is performed in this workflow.

> <warning-title>Feature-selection leakage</warning-title>
>
> The supplied workflow reproduces the attached analysis and selects features before splitting the data. For a publication-quality performance estimate, split first, learn feature selection only on the training partition, and apply the learned transformation to the held-out test partition. Selecting features on all samples can make test performance optimistic.
{: .warning}

> <hands-on-title>Select ten genes</hands-on-title>
>
> 1. {% tool [Feature Selection](toolshed.g2.bx.psu.edu/repos/bgruening/sklearn_feature_selection/sklearn_feature_selection/1.0.11.2) %}:
>    - *"Select a feature selection algorithm"*: `SelectKBest`
>    - *"Score function"*: `chi2`
>    - *"Number of top features"*: `10`
>    - *"Input type"*: `tabular`
>    - {% icon param-file %} *"Training samples dataset"*: `expression and labels`
>    - *"Does the dataset contain a header?"*: `Yes`
>    - *"Choose how to select data by column"*: `All columns but by header name`
>    - *"Header name"*: `label`
>    - {% icon param-file %} *"Target dataset"*: `expression and labels`
>    - *"Does the dataset contain a header?"*: `Yes`
>    - *"Choose how to select data by column"*: `Select columns by header name`
>    - *"Header name"*: `label`
> 2. Rename the output to `top 10 expression features`.
> 3. {% tool [Paste](Paste1) %} the `top 10 expression features` and `encoded labels`, using a tab delimiter.
> 4. Rename the result to `modeling table`.
>
{: .hands_on}

> <question-title></question-title>
>
> 1. Why is dimensionality reduction important here?
> 2. What assumption does the chi-squared score make about input values?
>
> > <solution-title></solution-title>
> >
> > 1. There are far more genes than samples. Reducing the number of inputs lowers computation and reduces the opportunity to model noise.
> > 2. Chi-squared feature selection requires non-negative values. The supplied expression matrix satisfies this requirement.
> >
> {: .solution}
>
{: .question}

# Make a shared train/test split

A stratified split preserves the representation of each cancer type and gives both models identical training and evaluation cases.

> <hands-on-title>Split the data</hands-on-title>
>
> 1. {% tool [Split Dataset](toolshed.g2.bx.psu.edu/repos/bgruening/sklearn_train_test_split/sklearn_train_test_split/1.0.11.2) %}:
>    - {% icon param-file %} *"Dataset containing input features"*: `modeling table`
>    - *"Does the dataset contain a header?"*: `Yes`
>    - *"Split mode"*: `Train and test split`
>    - *"Test size"*: `0.25`
>    - *"Random seed"*: `42`
>    - *"Shuffle"*: `Stratified shuffle`
>    - {% icon param-file %} *"Dataset containing class labels"*: `encoded labels`
>    - *"Column containing class labels"*: `1`
> 2. Rename the two outputs to `training data` and `test data`.
>
{: .hands_on}

# Train the classifiers

## Random Forest baseline

> <hands-on-title>Train and apply Random Forest</hands-on-title>
>
> 1. {% tool [Ensemble methods](toolshed.g2.bx.psu.edu/repos/bgruening/sklearn_ensemble/sklearn_ensemble/1.0.11.2) %}:
>    - *"Select a task"*: `Train a model`
>    - *"Select an ensemble method"*: `RandomForestClassifier`
>    - {% icon param-file %} *"Training samples dataset"*: `training data`
>    - Select all columns except the header name `label`
>    - {% icon param-file %} *"Target dataset"*: `training data`
>    - Select the column with header name `label`
>    - *"Number of trees"*: `100`
>    - *"Criterion"*: `gini`
> 2. Rename the fitted model to `Random Forest model`.
> 3. {% tool [Cut](Cut1) %} the first ten columns (`c1-c10`) from `test data` and rename the result `Random Forest test features`.
> 4. Run {% tool [Ensemble methods](toolshed.g2.bx.psu.edu/repos/bgruening/sklearn_ensemble/sklearn_ensemble/1.0.11.2) %} again:
>    - *"Select a task"*: `Load a model and predict`
>    - {% icon param-file %} *"Models"*: `Random Forest model`
>    - {% icon param-file %} *"Data"*: `Random Forest test features`
>    - *"Does the dataset contain a header?"*: `Yes`
>    - *"Prediction method"*: `predict`
>
{: .hands_on}

## TabPFN

TabPFN receives the labeled training table and the labeled test table. The test labels are retained only so the tool can return them alongside predictions; they are not prediction inputs.

> <hands-on-title>Run TabPFN classification</hands-on-title>
>
> 1. {% tool [Tabular data prediction using TabPFN](toolshed.g2.bx.psu.edu/repos/bgruening/tabpfn/tabpfn/7.0.0+galaxy1) %}:
>    - *"Task"*: `Classification`
>    - {% icon param-file %} *"Train data"*: `training data`
>    - *"Train data has a header"*: `Yes`
>    - {% icon param-file %} *"Test data"*: `test data`
>    - *"Test data has a header"*: `Yes`
>    - *"Test data has labels"*: `Yes`
>    - *"Model source"*: `Preinstalled model`
>    - *"Pretrained model"*: `tabpfn-v2.5-classifier-v2.5_default.ckpt`
>    - *"I certify that the model is used according to its license"*: `Yes`
>    - Request one GPU where Galaxy exposes job-resource controls.
> 2. Rename the prediction output to `TabPFN predictions`.
>
>    > <comment-title>Compute resources</comment-title>
>    >
>    > The accompanying workflow requests a GPU. Availability and the exact resource selector depend on the Galaxy server. Ask the server administrators if the pretrained model or GPU destination is not offered.
>    {: .comment}
>
{: .hands_on}

# Evaluate and compare the models

The true labels are column 11 of the test table. We extract the corresponding prediction column from each model and generate confusion matrices.

> <hands-on-title>Create the TabPFN confusion matrix</hands-on-title>
>
> 1. {% tool [Advanced Cut](toolshed.g2.bx.psu.edu/repos/bgruening/text_processing/tp_cut_tool/9.5+galaxy3) %} on `test data`:
>    - *"Operation"*: `Keep fields`
>    - *"List of fields"*: column `11`
>    - *"File has a header"*: `Yes`
> 2. Rename the output to `true test labels`.
> 3. Use {% tool [Advanced Cut](toolshed.g2.bx.psu.edu/repos/bgruening/text_processing/tp_cut_tool/9.5+galaxy3) %} with the same settings on `TabPFN predictions`, retaining column 11. Rename it `TabPFN predicted labels`.
> 4. {% tool [Machine Learning Visualization Extension](toolshed.g2.bx.psu.edu/repos/bgruening/ml_visualization_ex/ml_visualization_ex/1.0.11.2) %}:
>    - *"Select a plot type"*: `Classification confusion matrix`
>    - {% icon param-file %} *"Data with true class labels"*: `true test labels`
>    - Select the column with header name `label`
>    - {% icon param-file %} *"Data with predicted class labels"*: `TabPFN predicted labels`
>    - *"Predicted data has a header"*: `Yes`
>    - *"Plot title"*: `Confusion matrix between true and predicted labels`
>    - *"Plot format"*: `png`
>    - *"Color map"*: `Purples`
> 5. Rename the plot to `TabPFN confusion matrix`.
>
{: .hands_on}

> <hands-on-title>Create the Random Forest confusion matrix</hands-on-title>
>
> 1. {% tool [Cut](Cut1) %} column 11 (`c11`) from the Random Forest prediction output. Rename it `Random Forest predicted labels`.
> 2. Run {% tool [Machine Learning Visualization Extension](toolshed.g2.bx.psu.edu/repos/bgruening/ml_visualization_ex/ml_visualization_ex/1.0.11.2) %} with the same confusion-matrix settings:
>    - {% icon param-file %} *"Data with true class labels"*: `test data`; select header `label`
>    - {% icon param-file %} *"Data with predicted class labels"*: `Random Forest predicted labels`
>    - *"Predicted data has a header"*: `Yes`
>    - *"Plot format"*: `png`
>    - *"Color map"*: `Purples`
> 3. Rename the plot to `Random Forest confusion matrix`.
>
{: .hands_on}

> <question-title></question-title>
>
> 1. What does a confusion-matrix cell away from the diagonal represent?
> 2. Is the model with the largest diagonal count necessarily best for every cancer type?
> 3. Why must the two models use the same test partition?
>
> > <solution-title></solution-title>
> >
> > 1. It represents a sample whose true class (row) was assigned to a different predicted class (column).
> > 2. No. Overall correct predictions can hide poor recall for a smaller class. Inspect each row and consider per-class precision, recall, and F1 score for a fuller comparison.
> > 3. A shared test set removes variation caused by evaluating different samples, making differences attributable to the models rather than the split.
> >
> {: .solution}
>
{: .question}

# Run the complete workflow

The tutorial directory includes the workflow **Cancer prediction using gene expression data by TabPFN (no preprocessing)**. Import it through Galaxy's workflow menu, select the notebook, series file, and matrix file for its three inputs, and run it to reproduce the complete analysis.

> <details-title>Workflow input mapping</details-title>
>
> - `Ipython script` → `preprocess_expression_data.ipynb`
> - `GSE68086 series file` → `GSE68086 series file`
> - `GSE68086 matrix file` → `GSE68086 matrix file`
>
{: .details}

# Conclusion

You transformed a public platelet RNA-seq dataset into a labeled sample-by-gene table, selected ten informative genes, made a reproducible stratified split, and compared TabPFN with a Random Forest baseline. Confusion matrices show which cancer types each model separates and which it confuses. For rigorous downstream benchmarking, move feature selection inside the training partition and add repeated cross-validation or an independent validation cohort.

