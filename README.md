
## "You too Brutus! Trapping Hateful Users in Social Media: Challenges, Solutions & Insights [Accepted at ACM Conference on Hypertext and Social Media 2021] 


------------------------------------------
**Contributions** :volcano:	
------------------------------------------

* We crate a **new dataset** of hateful and non-hateful users from Gab and try to detect hateful users using GNN based model. 
* We explore several supervised, unsupervised and **semi supervised machine learning models**, including the state-of-the-art deep learning models, to classify users as hateful and nonhateful.
* We also notice that structural signatures learnt from a network are transferable in a zero shot setting to an unseen dataset.



**Please cite our paper in any published work that uses any of these resources.**

~~~bibtex
@inproceedings{das2021you,
  title={You too brutus! trapping hateful users in social media: Challenges, solutions \& insights},
  author={Das, Mithun and Saha, Punyajoy and Dutt, Ritam and Goyal, Pawan and Mukherjee, Animesh and Mathew, Binny},
  booktitle={Proceedings of the 32nd ACM Conference on Hypertext and Social Media},
  pages={79--89},
  year={2021}
}
~~~

# Links (optional).
gab_dataset : "https://zenodo.org/record/5140191#.YQBGhI4zY2w"

twitter_dataset : "https://www.kaggle.com/manoelribeiro/hateful-users-on-twitter"


------------------------------------------
**Folders' Description** :open_file_folder:	
------------------------------------------
~~~
./Dataset                               --> Please Keep the relavant dataset in this folder by downloading them.
./Preprocessing and EmbCreation	    	--> Contains codes to preporcess the datasets and create Doc2vec Embedding
./GABModels                             --> Contains all the GNN based models to train the Gab dataset.
./TwitterModels                         --> Contains all the GNN based models to train the Twitter dataset.
./Cross-Platform                        --> Contains code to try out the zero-shot setting on Cross-Platform.
~~~



----------------------------------------------------------
**Ethics note :eye_speech_bubble:**
----------------------------------------------------------

We only analyzed publicly available data. We did not make any attempts to track users across sites or deanonymize them. Also, taking into account user privacy, we anonymized the users information such as user name, user id etc.
