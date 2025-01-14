Argumentative Sentences in Spoken Language  (version 1.0)
--------------------------------------------------------
The file contains 700 sentences that were extracted from ASR of debate speeches over controversial topics.
A sentence is annotated as positive if it contains an argument for the given topic.

Structure of the CSV file
--------------------------

The CSV file presents 10 columns:
1. sentence_id
2. sentence
		a sentence extracted from a spoken debate speech
3. context
		a segment containing the sentence before the extracted sentence, the extracted sentences and the sentence that follows it
4. topic
		the topic of the debate
5. label
		1 if the sentence contains an argument, 0 otherwise, define by the majority vote of the annotators. 
6. #positive
	  number of annotators that answered that the sentence contains an argument
7. #negative
		number of annotators that answered that the sentence does not contains an argument.
8. sentence_internal_id
		id for authors internal use.
9. val	TRUE if this sentence is part of the validation set, FALSE otherwise.
10. test	TRUE if this sentence is part of the test set, FALSE otherwise.


Examples
--------
Here are 3 lines taken from argumentative_sentences_in_spoken_language_with_split.csv:
sentence_id							sentence                                            context                                              topic 						label 	#positive #negative sentence_internal_id	val	test
0            This has worked partially and very slowly and ...  But unlike proposition we do not think that th...                      We should ban beauty contests     0         0         3                  834		FALSE	TRUE	
1            So let's start with that about why this is ult...    So let's start with that about why this is u...                      We should ban anonymous posts     0         1         2                 1375		FALSE	TRUE
2            We think that governments in general should on...  So we ought not do it  We think that governmen...                      We should ban beauty contests     1         2         1                  877		FALSE	TRUE


Annotation process
------------------
Each sentence was annotated by 3 professional annotators.
The sentences were presented with their context and annotators had to decide for each sentence if it contains an argument.

The guidelines provided to the annotators:
In the following task you are given a part of a transcription of a spoken speech delivered over a controversial topic.
Note, the transcription is often done automatically, hence may contain errors (such as wrong transcription of words, bad split of the speech into sentences).
Try to figure out what the speaker really said and base your decisions on that.

A sentence is given with its context in the speech. For this sentence you should determine whether it contains an argument for the given topic.

An argument is a piece of text which directly supports or contests the given topic.
Note: having a clear stance towards the topic (either pro or against) is a critical prerequisite for a piece of text to be an argument.


Licensing and copyright
-----------------------
The dataset is released under the licensing and copyright terms mentioned in the
IBM Debater Datasets webpage: http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml


Citation
--------           
If you use the data, please cite the following publication:
Unsupervised Expressive Rules Provide Explainability and Assist Human Experts Grasping New Domains
Eyal Shnarch, Leshem Choshen, Guy Moshkowich, Noam Slonim and Ranit Aharonov
Findings of the Association for Computational Linguistics: EMNLP 2020