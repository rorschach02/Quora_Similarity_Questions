# Quora Question Pair Similarity


Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.  so main aim of project is that predicting whether pair of questions are similar or not. This could be useful to instantly provide answers to questions that have already been answered.

<br/>

### Problem Statement :

Identify which questions asked on Quora are duplicates of questions that have already been asked.

### Real world/Business Objectives and Constraints :

   - The cost of a mis-classification can be very high.
   - You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
   - No strict latency concerns.
   - Interpretability is partially important.

### Performance Metric:

   - log-loss 
   - Binary Confusion Matrix

### Data Overview:
Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate. <br/>
Total we have 404290 entries. Splitted data into train and test with 70% and 30%.

i derived some features from questions like no of common words, word share and some distances between questions with the help of word vectors. will discuss those below. You can check my total work.

### Basic Analysis:

- ##### Distribution of data points among output classes  
![1](https://user-images.githubusercontent.com/46763031/149611108-d8766543-9328-4358-b036-0c88e65b7bab.PNG)

- ##### Number of unique questions
![2](https://user-images.githubusercontent.com/46763031/149611124-01498a58-e4bc-4c3d-ab1b-0892e1ea1e32.PNG)

- ##### Number of occurrences of each question - Log Histogram
![3](https://user-images.githubusercontent.com/46763031/149611145-adef421a-07d9-4e03-b9cc-281fe006f78a.PNG)

### Feature Extraction:
- ##### Basic Features - Extracted some features before cleaning of data as below.
  - <b>freq_qid1</b> = Frequency of qid1's
  - <b>freq_qid2</b> = Frequency of qid2's
  - <b>q1len</b> = Length of q1
  - <b>q2len</b> = Length of q2
  - <b>q1_n_words</b> = Number of words in Question 1
  - <b>q2_n_words</b> = Number of words in Question 2
  - <b>word_Common</b> = (Number of common unique words in Question 1 and Question 2)
  - <b>word_Total</b> =(Total num of words in Question 1 + Total num of words in Question 2)
  - <b>word_share</b> = (word_common)/(word_Total)
  - <b>freq_q1+freq_q2</b> = sum total of frequency of qid1 and qid2
  - <b>freq_q1-freq_q2</b> = absolute difference of frequency of qid1 and qid2
- ##### Advanced Features - Did some preprocessing of texts and extracted some other features. i am giving some definitions which are used below. `Token`- You get a token by splitting sentence by space  ,  `Stop_Word` - stop words as per NLTK, `Word `-A token that is not a stop_word.
  - <b>cwc_min</b> = common_word_count / (min(len(q1_words), len(q2_words)) 
  - <b>cwc_max</b> = common_word_count / (max(len(q1_words), len(q2_words)) 
  - <b>csc_min</b> = common_stop_count / (min(len(q1_stops), len(q2_stops)) 
  - <b>csc_max</b> = common_stop_count / (max(len(q1_stops), len(q2_stops)) 
  - <b>ctc_min</b> = common_token_count / (min(len(q1_tokens), len(q2_tokens)) 
  - <b>ctc_max</b> = common_token_count / (max(len(q1_tokens), len(q2_tokens)) 
  - <b>last_word_eq</b> = Check if Last word of both questions is equal or not (int(q1_tokens[-1] == q2_tokens[-1]))
  - <b>first_word_eq</b> = Check if First word of both questions is equal or not (int(q1_tokens[0] == q2_tokens[0]) )
  - <b>abs_len_diff</b> = abs(len(q1_tokens) - len(q2_tokens))
  - <b>mean_len</b> = (len(q1_tokens) + len(q2_tokens))/2
  - <b>fuzz_ratio</b> = How much percentage these two strings are similar, measured with edit distance.
  - <b>fuzz_partial_ratio</b> = if two strings are of noticeably different lengths, we are getting the score of the best matching lowest length substring.
  - <b>token_sort_ratio</b> = sorting the tokens in string and then scoring fuzz_ratio.
  - <b>longest_substr_ratio</b> = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))
- ##### Extracted Tf-Idf features for this combained question1 and question2 and got 1,2,3 gram features with Train data. Transformed test data into same vector space. 
- ##### Got [Word Movers Distance](http://proceedings.mlr.press/v37/kusnerb15.pdf) with pretrained glove word vectors. 
- ##### From Pretrained glove word vectors got average word vector for question1 and question2. With this avg word vector got below distances. 
  - <b>Cosine distance</b>
  - <b>Cityblock distance</b>
  - <b>Canberra distance</b>
  - <b>Euclidean distance</b>
  - <b>Minkowski distance</b>

### Some Features analysis and visualizations:
- ##### word_share - We can check from below that it is overlaping a bit, but it is giving some classifiable score for disimilar questions.
  ![4](https://user-images.githubusercontent.com/46763031/149611248-a470aff0-197a-49f6-be63-2dbdebc24112.PNG)
- ##### Word Common - it is almost overlaping.
   ![5](https://user-images.githubusercontent.com/46763031/149611255-e2781814-c11b-4467-8bd1-f3ebb8a3e40d.PNG)

- ##### Bivariate analysis of features 'ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio'. We can observe that we can divide duplicate and non duplicate with some of these features with some patterns. 
   ![6](https://user-images.githubusercontent.com/46763031/149611294-a23babae-7bb6-4649-b0ba-894070111295.PNG)
- ##### Distribution of the token_sort_ratio
 ![7](https://user-images.githubusercontent.com/46763031/149611337-21e6e228-4165-49df-91f8-456daae606b1.PNG)

- ##### Distribution of the fuzz_ratio
  
![8](https://user-images.githubusercontent.com/46763031/149611338-3fcf96cc-02a1-4985-befc-0fc37d0dc286.PNG)

- ##### Wordcloud for similar questions
![9](https://user-images.githubusercontent.com/46763031/149611382-0538bad6-d06c-4240-a097-dac44091f347.PNG)

- ##### Wordcloud for dissimilar questions
![10](https://user-images.githubusercontent.com/46763031/149611385-4f5f11ee-e633-4cc5-8561-c68fcb451f3b.PNG)

### TSNE for Dimentionality reduction:
Using TSNE for Dimentionality reduction for 15 Features(Generated after cleaning the data) to 3 dimention

![11](https://user-images.githubusercontent.com/46763031/149611406-4857f5df-1df9-497a-b5f9-ee23f0dedda4.PNG)

### Machine Learning Models:
  Various models has been tried like **Logistic Regression, Linear SVM and GBDT** with **BoW, TF-IDF and W2V** vectorizer along with hand-crafted features. But out of all these   **GBDT with TFIDF vectorizer** gave the best performance.
   For all other models training, calibration and tunning look at the notebook,

   Performance Summary: 
| Model         | Test Log Loss |
| ------------- | ------------- |
| Random Model  |  0.887242646958  |
| Logistic Regression  |  0.520035530431  |
| Linear SVM  |  0.489669093534 |
| XGBoost  |  0.357054433715  |


##### References:
1. https://www.kaggle.com/c/quora-question-pairs 
2. https://www.kaggle.com/c/quora-question-pairs/discussion
3. Applied AI Course
4. https://github.com/seatgeek/fuzzywuzzy#usage , https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/








