NAME: IBM Debater(R) - SurveyKP

VERSION: v1

RELEASE DATE: December 5, 2023

DATASET OVERVIEW

15,189 (sentence, key point) pairs labeled as matching/non-matching/undecided.
The sentences were sampled from open-ended responses to the 2016-2017 Austin Community Survey. 
The key points were automatically extracted by our KPA system from the entire survey, with minor manual edits.
The stance of each pair (whether the feedback is positive or negative/suggestion) is also indicated.

The dataset is released under the following licensing and copyright terms:
• (c) Copyright IBM 2023. Released under Community Data License Agreement – Sharing, Version 1.0 (https://cdla.io/sharing-1-0/).

The dataset is described in the following publication (referred to as the "Municipal" test set): 

• Welcome to the Real World: Efficient, Incremental and Scalable Key Point Analysis. Lilach Eden, Yoav Kantor, Matan Orbach, Yoav Katz, Noam Slonim, Roy Bar-Haim. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track. 2023

Please cite this paper if you use the dataset.

CONTENTS

The CSV file, SurveyKP_dataset.csv, contains the following columns for each (sentence, key point) pair:
1. sentence
2. key_point	
3. stance: 1 (pro) / -1 (con)
4. label: 1 (matching)/ 0 (non-matching) / -1 (undecided)

Note: In the experiments described in the paper, the undecided examples were excluded.