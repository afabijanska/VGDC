# Viral Genome Deep Classifier (VGDC)

## INTRO

This directory contains the source code of Viral Genome Deep Classifier (VGDC) -a tool for an automatic virus subtyping, which employs a deep convolutional neural network. 

The source code may be used for non-commercial research provided you acknowledge the source by citing the following paper:</p>

<ul>
<li><b>Fabijańska A.</b>, Grabowski S.: <i>Viral Genome Deep Classifier</i>, IEEE Access, vol. 7, pp. 81297-81307, 2019, doi:10.1109/ACCESS.2019.2923687 
<li>
</ul>
  
<pre><code>@article{Fabijanska2019,<br>
  author  = {Anna Fabija\'{n}ska and Szymon Grabowski}, <br>
  title   = {Viral Genome Deep Classifier},<br>
  journal = {IEEE Access},<br>
  volume  = {7},<br>
  number  = {},<br>
  pages   = {81297-81307},<br>
  year 	  = {2019},<br>
  note 	  = {},</br>
  issn 	  = {2169-3536},<br>
  doi 	  = {10.1109/ACCESS.2019.2923687}, <br>
  url 	  = {https://doi.org/10.1109/ACCESS.2019.2923687}<br>
}</code></pre>

## PREREQUISITES

The method was implemented in Python 3.6 with the use of Keras library running at the top of TensorFlow

## DIRECTORY CONTENT RELATED TO VGCD

- parseGenomeDengue.py
  a parser of fasta files containing genomes of Dengue virus; takes a fasta file as input and saves list of pairs ('label','genome') in a pickle format; location of a fasta file and location of the output list are defined in a configuration file  

- parseGenomeHepatisB.py
  a parser of fasta files containing genomes of Hepatitis B virus; takes a fasta file as input and saves list of pairs ('label','genome') in a pickle format; location of a fasta file and location of the output list are defined in a configuration file   

- parseGenomeHepatisC.py
  a parser of fasta files containing genomes of Hepatitis C virus; takes a fasta file as input and saves list of pairs ('label','genome') in a pickle format; location of a fasta file and location of the output list are defined in a configuration file   

- parseGenomeHiv-1.py 
  a parser of fasta files containing genomes of HIV-1 virus; takes a fasta file as input and saves list of pairs ('label','genome') in a pickle format; location of a fasta file and location of the output list are defined in a configuration file   

- parseGenomeInfluenza.py
  a parser of fasta files containing genomes of Influenza A virus; takes a fasta file as input and saves list of pairs ('label','genome') in a pickle format; location of a fasta file and location of the output list are defined in a configuration file   

- nFoldValidation.py
  creates training/testing folds for N-fold cross validation; takes a list of pairs ('label','genome') generated by a parser as input, generates training/testing folds randomly and saves them in a pickle format; a path to input data is defined in a configuration file; the resulting data is saved in N pairs of files trainK.p/testK.p where k=1, 2, ..., N  

- train_CNN.py
  defines the architecture of VGDC, trains the CNN behind it and saves resulting weights in a h5 format; path to training data is defined in a configuration file; paths to files containing weights and a model in JSON format are also defined in a configuration file

- predict_and_evaluate_CNN.py
  reads resulting weights from h5 file, performs prediction and evaluates the results; path to testing data is defined in a configuration file; path to file containing weights is also defined in a configuration file 

- config.txt
  a configuration file, defines paths to fasta file, training/testing data and training parameters 

- helpers.py 
  some helper functions

## HOW TO RUN VGCD

1. Edit a configuration file
2. Run selected parser
3. Run nFoldValidation.py
4. Run train_CNN.py
5. Run predict_and_evaluate.py

## HOW TO RUN COMPETITIVE METHODS

How to run C-Measure based classification.

The data for a given virus are in one of the subdirectories of
http://an-fab.kis.p.lodz.pl/genomes/trained_CNN/

Consider, as an example, the "Hiv-1 (12 classess)" subdirectory.

1. From http://an-fab.kis.p.lodz.pl/genomes/trained_CNN/Hiv-1%20(12%20classess)/
download all train*.p and all test*.p files (10 files in total, which correspond 
to the train and test files in a 5-fold cross validation).

2. Invoke
python c_measure.py > hiv-1_12classes.log
(where Python 3.x is used; we tested it with Python 3.6).

3. After a couple of minutes, or hours in some cases, see the created log file; 
its last lines will contain total results over 5 folds.


How to perform PPMT-based classification.

1. Download http://compression.ru/ds/ppmtrain.rar and extract it
(it contains sources, 2 Windows executables (PPMTrain.exe and PPMd1/PPMd1.exe) and READ_ME.TXT).
We tested it under Windows only.

2. From http://an-fab.kis.p.lodz.pl/genomes/trained_CNN/Hiv-1%20(12%20classess)/
download all train*.p and all test*.p files (10 files in total, which correspond
to the train and test files in a 5-fold cross validation). 
Store them in one directory, together with the PPM*.exe files.
Let ppmCompressionClassifier.py be also in this directory.

3. Invoke
python ppmCompressionClassifier.py > hiv-1_12classes.log
(where Python 3.x is used; we tested it with Python 3.6).

4. After a couple of minutes, or hours in some cases, see the created log file 
(check its last lines for a summary).




