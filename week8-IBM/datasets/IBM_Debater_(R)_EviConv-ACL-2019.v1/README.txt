Evidence Convincingness (version 1.0)
-------------------------------------------------

The data set contains 1,844 confirmed evidence taken from the data set of Shanrch et al. (2018). Out of these pieces of evidence 5,697 pairs were sampled. Each pair was annotated for the question of which evidence is more convincing.

The data set is split into 4,319 pairs for train and 1,378 for test (each is provided as a csv file).
Train set includes 48 topics and test set includes 21 other topics (i.e., no topic overlap between the two sets).

This file is only the README, the data set itself is available on the
IBM Debater Datasets webpage: http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml
and is comprised of 4 files: this readme, train.csv and test.csv, and experts_reasons.csv.


Structure of the train and test files
-------------------------------------

Each file presents 15 columns:

1. topic
		The debatable topic serving as context for the evidences.
2. evidence_1
		The first evidence of the pair
3. evidence_2
		The second evidence of the pair
4. label
		1 if evidence_1 is the more convincing one, 2 if evidence_2 is the more convincing
5. acceptance_rate
		the fraction of labelers which choose the final label 
6. evidence_1_stance
		PRO for evidence supporting the topic, CON for evidence opposing the topic
7. evidence_2_stance
		same as above for evidence 2
8. evidence_1_detection_score
		the score of the detection model for evidence 1, described in Section 5.2 in the paper
9. evidence_2_detection_score
		same as above for evidence 2
10. evidence_1_id
		an internal id of an evidence
11. evidence_2_id
		same as above for evidence 2
12. evidence_1_wikipedia_article_name
		The wikipedia article name from which evidence 1 was extracted
13. evidence_2_wikipedia_article_name
		same as above for evidence 2
14. evidence_1_wikipedia_url
		The wikipedia article's URL from which evidence 1 was extracted
15. evidence_2_wikipedia_url
		same as above for evidence 2

		
Examples
--------

Here are 3 lines taken from train.csv, following the line format:
topic,evidence_1,evidence_2,label,acceptance_rate,evidence_1_stance,evidence_2_stance,evidence_1_id,evidence_2_id
We should end affirmative action,"In February 2007, Sarkozy appeared on a televised debate on TF1 where he expressed his support for affirmative action and the freedom to work overtime.","Williams believes programs such as affirmative action and minimum wage laws set up to aide minorities have, in fact, been harmful to them and stifled their ability to advance in society.",2,1.00,CON,PRO,2469,2478
We should subsidize condoms,"In 2009, Lewis strongly criticized Pope Benedict XVI's assertion that condom use only makes the AIDS crisis worse [REF].","Green said that according to the ""best studies,"" condoms makes people take wilder sexual risks, thus worsening the spread of the disease.",2,0.63,PRO,CON,1743,1750
We should legalize prostitution,"The appellants' argument then, more precisely stated, is that in criminalizing so many activities surrounding the act itself, Parliament has made prostitution de facto illegal if not de jure illegal."",","Feminists who hold such views on prostitution include Kathleen Barry, Melissa Farley,[REF][REF] Julie Bindel,[REF][REF] Sheila Jeffreys, Catharine MacKinnon[REF] and Laura Lederer;[REF] the European Women's Lobby has also condemned prostitution as ""an intolerable form of male violence"" [REF].",1,0.71,CON,CON,1564,1608


The columns of the experts_reasons.csv are similar to those of train.csv and test.csv files, with a additional 'reason' column.
The reasons in this new column were given by our in-house expert labelers who were asked to supply factors for commonly deciding the preference between two pieces of evidence. For more information, see section 6.2 in the paper.

Annotation process
------------------

Each evidence pair was annotated by 10 crowd-sourced annotators.

The guidelines provided to the annotators:
In this task you are presented with pairs of evidence texts for a given topic of debate. For each pair of candidates, you should pick the one you find more convincing. The candidates may support opposite sides of the topic, this is irrelevant for this task. You need to choose the one which is stronger for its respective stance.

Be decisive: in a conversation about the topic, where you can only give a single evidence out of these two - which one would you rather use (to either support or contest the topic, it doesn't matter)?

Some evidence texts include the [REF] sign. It indicates that this evidence was taken from a written source (the [REF] sign replaces the reference text to the source).


Licensing and copyright
-----------------------
The dataset is released under the licensing and copyright terms mentioned in the
IBM Debater Datasets webpage: http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml


Citation
--------           
If you use the data, please cite the following publication:

	Are You Convinced? Choosing the More Convincing Evidence with a Siamese Network   	
	Martin Gleize, Eyal Shnarch, Leshem Choshen, Lena Dankin, Guy Moshkowich, Ranit Aharonov, Noam Slonim
	Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019 


The confirmed evidence were taken from Shanrch et al. (2018):
	Will it Blend? Blending Weak and Strong Labeled Data in a Neural Network for Argumentation Mining
	Eyal Shnarch, Carlos Alzate, Lena Dankin, Martin Gleize, Yufang Hou, Leshem Choshen, Ranit Aharonov and Noam Slonim.
	Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, 2018
