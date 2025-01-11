<h3>Deliverable Layout</h3>

  

<h4>Data</h4>

  

[data_folder](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/tree/master/data)

  

-  [dataset.csv](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/data/dataset.csv)

  

*Notes: cleaned dataset*

  

-  [preprocesspyFinal.py](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/data/preprocesspyFinal.py)

  

*Notes: pre-processing code for cleaning documents*

  

-  [Facebook Public Posts for Cornell Project 2016-2020.xlsx](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/data/Facebook%20Public%20Posts%20for%20Cornell%20Project%202016-2020.xlsx)

  

*Notes: original data from Facebook*

  

-  [Twitter Public Posts for Cornell Project 2016-2020.xlsx](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/data/Twitter%20Public%20Posts%20for%20Cornell%20Project%202016-2020.xlsx)

  

*Notes: original data from Twitter*

  

-  [contraction_map.json](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/data/contraction_map.json)

  

*Notes: Resource for cleaning data*

 

  

<h4>Code</h4>

  

[code folder](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/tree/master/code)

  

-  [Main](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/tree/master/code/Main)

-  [Li.ipynb](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/code/Main/Li.ipynb)

*Notes: Using the keras_dataset with sentiment label, conducted topic mining with K-means, and then trend analysis with word2vec similarity, inicluding chart to trend analysis visualization.*

- [txtblob_lda_w2v_actionability.ipynb](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/code/Main/txtblob_lda_w2v_actionability.ipynb)
*Notes: Generated sentiment scores from textblob and distinguish results according to polarity. Performed topic mining with LDA on cases with negative polarity. Built word2vec and keyedvector models based on sentiment data.Classified samples under topics with word2vec similarity.Visualized results and performed trend analysis. Built a demo actionability classifier.*


-  [Extra](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/tree/master/code/Extra)

  

​ [textblob_label.py](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/code/Extra/textblob_label.py)

  

*Notes: using textblob as sentiment classifier to label each documents sentiment by polarity, saving textblob_dataset.csv*

  

​ [auto_label_IBM.py](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/code/Extra/auto_label_IBM.py)

  

*Notes: using IBM Waston as sentiment classifer to label 30k randomly selectly documents sentiment, editing on the textblob_datasetcsv, and saving IBM_30k_dataset.csv*

  

​ [Keras.ipynb](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/code/Extra/Keras.ipynb)

  

*Notes: Using word2Vec and Keras on TensorFlow to train a sentiment classifier based on the IBM Label, this is a demo testing for the development of our future relevance classifer. Additonally, we use the model we train to predict the sentiment of the rest of tweets which are not labeled by IBM.*

  

​ [keras_dataset.csv](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/code/Extra/keras_dataset.csv)

  

Notes: the results of the Keras model we built, including sentiment polarity from textblob and the label from Keras sentmentt classifier.


[dataset_vador_textblob.csv](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/code/Extra/dataset_vador_textblob.csv)

*Notes: Tweets labeled by sentiment scores using vendor and textblob library.*

[similarity_data.csv](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/code/Extra/similarity_data.csv)

*Notes: Similarity rate between each sample and each topic, labeled with topic of highest similarity rate.*

[w2v_model.model](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/code/Extra/w2v_model.model)  & [word_vectors.model](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/code/Extra/word_vectors.model)

*Notes: Two models that represents text as feature vectors and can be processed the similar way as deep neural networks.*

[lda_model.model](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/code/Extra/lda_model.model)

*Notes: LDA model for topic mining*

[vader_label.ipynb](https://github.coecis.cornell.edu/info5901-2020sp/navy-federal-project/blob/master/code/Extra/vader_label.ipynb)

*Notes:  Labeled samples with sentiment scores using Vader*

