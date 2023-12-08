# Final Project: Anime Recommendation System

#### Prepared by: Abhinav Goel

This is the final project of a recommendation system to fulfill the submission requirements for Dicoding. The project builds a content-based filtering model that can determine the top-N anime recommendations for users.

## Project Domain

### Background

Animation is the process of creating motion effects on an object through continuous changes in a collection of images over a specific period. This process also adds sound effects, emotions, and characters according to the story, resulting in objects that appear to come to life. Anime does not have a specific meaning and is a term used by the Japanese to refer to animation, whether it is made in Japan or not. However, for people outside Japan, the term "anime" is used to distinguish Japanese-made animation from animation produced in America due to the distinctive characteristics of animation in these two countries.

<br>

<div><img src="https://user-images.githubusercontent.com/107544829/190916199-7df48b2b-556b-41da-a0f1-ae37f91c3aee.png" width="1000"/></div>

[Referensi gambar](https://wall.alphacoders.com/big.php?i=546902)

<br>

Anime is extremely popular both within and outside Japan, leading to the existence of numerous online streaming sites and systems that enable users to watch anime from anywhere. However, the current abundance of available anime can sometimes pose a challenge for users to find anime that aligns with their preferences. This issue may also arise due to limited descriptions and user reviews.

In response to this problem, this research aims to provide anime recommendations to users based on the similarity of their preferences to previously watched anime. It is anticipated that by offering suitable anime recommendations, users can find entertainment more easily, spend more time within the system, and generate benefits for the developers as providers of the system.

References:

1. [Anime and the Lifestyle of University Students](https://repository.uinjkt.ac.id/dspace/bitstream/123456789/45316/1/Ida%20Aisyah.pdf)

2. [Anime Recommendation with Latent Semantic Indexing Based on Synopsis Genre](https://www.researchgate.net/publication/274712918_Rekomendasi_Anime_dengan_Latent_Semantic_Indexing_Berbasis_Sinopsis_Genre)

## Business Understanding

This project is built for a company with the following business characteristics:

- A company that develops a website or online anime streaming system.
- A company that develops a recommendation and information website for anime.

### Problem Statement

1. Can the system provide recommendations without input from new users?
2. Based on recently liked anime by users, how can we create a list of anime recommendations using the content-based filtering approach?

### Goals

1. Display a list of top anime recommendations for new users.
2. Generate a list of anime recommendations based on recently liked anime using the content-based filtering approach.

### Solution Statement

1. Analyze data by performing univariate analysis and multivariate analysis. Understanding the data can also be done through visualization. This stage can accomplish Goal 1.
2. Prepare data for use in building the model.
3. Develop the model using the content-based filtering approach and evaluate it. This stage can accomplish Goal 2.

## Data Understanding & Preprocessing

The dataset used in this project is a list of anime titles with characteristics such as the number of fans and average ratings from users. This dataset can be downloaded from [Kaggle: Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database).

Here is information about the dataset:
<br>

There are two datasets, namely the anime dataset containing each anime title along with its genre, type, episodes, average rating, and the number of community members for that anime. There is also the anime_rating dataset containing user ratings for each anime.

1. Anime Dataset
   - The dataset is in CSV format.
   - The dataset has 12,294 samples with 7 features.
   - The dataset has 1 float(64) feature, 2 int64 features, and 4 object features.
   - There are missing values in the dataset.

2. Anime Rating Dataset
   - The dataset is in CSV format.
   - The dataset has 7,813,737 samples with 3 features.
   - The dataset has 1 int64 feature.
   - There are no missing values in the dataset.

### Variables in the Dataset

1. Anime Dataset
   - `anime_id` = Unique ID for each anime.
   - `name` = Anime title.
   - `genre` = Anime genre.
   - `type` = Type of anime, such as TV, OVA, etc.
   - `episodes` = Number of episodes for each anime.
   - `rating` = Average rating for each anime based on the number of users who gave ratings.
   - `members` = Number of community members for each anime.

2. Anime Rating Dataset
   - `user_id` = Unique ID for each user.
   - `anime_id` = ID of the anime rated by the user.
   - `rating` = Rating given by the user.

### Data Preprocessing

1. Remove missing values from the anime dataset.
2. Remove duplicate samples from the anime rating dataset.
3. Remove symbols from anime titles.

### Univariate Analysis

Univariate Analysis involves analyzing each feature separately.

#### Analysis of each attribute in the anime dataset

|       | anime_id | rating   | members    |
|-------|----------|----------|------------|
| count | 12017.00 | 12017.00 | 12017.00   |
| mean  | 13638.00 |     6.48 | 18348.88   |
| std   | 11231.08 | 1.02     | 55372.50   |
| min   | 1.00     | 1.67     | 12.00      |
| 25%   | 3391.00  | 5.89     | 225.00     |
| 50%   | 9959.00  | 6.57     | 1552.00    |
| 75%   | 23729.00 | 7.18     | 9588.00    |
| max   | 34519.00 | 10.00    | 1013917.00 |

The anime dataset has the lowest anime rating of 1.67 and the highest rating of 10, with an average rating of 6.48. This dataset also has the lowest number of anime community members at 12 and the highest at 1,013,917, with an average of 18,348. The significant difference between the minimum and maximum values of the number of anime community members is expected due to some anime being very popular while others are not.

#### Analysis of Each Numerical Attribute in the anime_rating Dataset

|       | user_id    | anime_id   | rating     |
|-------|------------|------------|------------|
| count | 7813736.00 | 7813736.00 | 7813736.00 |
| mean  |   36727.96 |    8909.07 | 6.14       |
| std   | 20997.95   | 8883.95    | 3.73       |
| min   | 1.00       | 1.00       | -1.00      |
| 25%   | 18974.00   | 1240.00    | 6.00       |
| 50%   | 36791.00   | 6213.00    | 1552.00    |
| 75%   | 54757.00   | 14093.00   | 9.00       |
| max   | 73516.00   | 34519.00   | 10.00      |

The anime rating dataset has the lowest user-given rating for an anime as -1, and the highest rating is 10. A rating of -1 indicates that the user watched the anime but did not provide a rating. Samples where users did not provide a rating will be excluded from further analysis and are removed using the following code:

```python
anime_rating = anime_rating[~(anime_rating.rating == -1)]
```

#### Analysis of Categorical Feature "Genre" in the Anime Dataset

![Genre Analysis](https://user-images.githubusercontent.com/107544829/190631304-e27ab156-08a5-474f-9d7d-52fe2e58112f.png)

The anime dataset has a multitude of unique genres. It is noticeable that one anime title can have multiple genres or only one genre. This variety in genre assignment is common in anime.

#### Analysis of Categorical Feature "Type" in the Anime Dataset

![Type Analysis](https://user-images.githubusercontent.com/107544829/190632721-2f1bbc4f-8567-4cc9-b49f-2c0a5e28beb1.png)

Insights:

- 30.52% of anime are aired on TV.
- 27.33% of anime are in the form of OVA.
- 18.80% of anime are in the form of movies.

"Movie" refers to anime presented in the form of a film. Original Video Animation (OVA) refers to anime released physically (CD, DVD, HD-DVD, Blu-ray, etc.) without TV broadcasting. Original Net Animation (ONA) is anime released first through the internet. Specials are anime episodes with a short duration, usually not related to the original storyline or provided as fan service.

Reference: [Pengertian Episode OVA,ONA,OAD dan SPESIAL Dalam Anime](https://binarycode100.wordpress.com/2018/12/08/pengertian-episode-ovaonaoad-dan-spesial-dalam-anime/).

#### Analysis of the Distribution of Average Anime Ratings

![Average Rating Distribution](https://user-images.githubusercontent.com/107544829/190857442-0b12b379-067e-4595-9481-1cd20323f4c0.png)

Most anime have average ratings spread from 4 to 8.

#### Analysis of the Distribution of User Ratings

![User Rating Distribution](https://user-images.githubusercontent.com/107544829/190857570-e8c4fd5e-994c-4519-9792-dca9b66c30db.png)

Most user ratings are spread from 6 to 10.

### Multivariate Analysis

Multivariate Analysis shows the relationship between two or more features in the data.

#### Top 10 Largest Anime Communities

![Top Anime Communities](https://user-images.githubusercontent.com/107544829/190857731-c2932dff-c16c-4c4c-af47-29e59bc5de8a.png)

The anime "Death Note" has the highest community members, followed by "Shingeki no Kyojin" and "Sword Art Online." This information can be used by the system developers to recommend popular anime to users. The number of anime community members indicates that the anime is quite popular among users.

#### Top 10 Anime Based on Average Anime Ratings

![Top Anime by Average Rating](https://user-images.githubusercontent.com/107544829/190858010-98b51998-dde0-4dc2-b978-9284f8d2dc4c.png)

The anime "Taka no Tsume 8: Yoshida-kun no X-Files" has the highest average rating, followed by "Spoon-hime no Swing Kitchen" and "Mogura no Motoro." This information tends to be biased for recommendations because the average rating is influenced by the number of users who gave ratings. For example, an anime X may have a high average rating, but only 3 users gave ratings.

#### Top 10 Anime Based on User Rating Contributions

![Top Anime by User Rating Contributions](https://user-images.githubusercontent.com/107544829/190858357-7be813d7-67e6-4283-899c-6af270de3fb6.png)

The anime "Death Note" contributes the most user ratings, followed by "Sword Art Online" and "Shingeki no Kyokin." This information can be used by the system developers to recommend popular anime to users. This is because the more rating contributions, the more users watched that anime (popularity).

## Content-Based Filtering Model & Result

### Content-Based Filtering

The system built by this project is a simple recommendation system based on anime genre using content-based filtering.

A content-based recommendation system suggests content that is similar to what a user has liked before. If two pieces of content have similar or nearly similar characteristics, they can be considered similar.

For example, in an anime recommendation system, if a user likes the anime "Jujutsu Kaisen," the system can recommend other action genre anime.

![TF-IDF Matrix](https://user-images.githubusercontent.com/107544829/190860242-6e0e9d61-e54f-46d0-930e-415e73cebab0.png)

#### TF-IDF

TF-IDF (Term Frequency - In

<br>

|                                                                | Kisaku Spirit |
|----------------------------------------------------------------|---------------|
| Hourou Musuko                                                  | 0.000000      |
| Mama Puri!?                                                    | 1.000000      |
| Dark Blue                                                      | 1.000000      |
| Anime Tenchou                                                  | 0.000000      |
| Chocolate Underground                                          | 0.000000      |
| Anata dake Konbawa                                             | 1.000000      |
| Bishoujo ANimerama: Gokkun Doli - Choujigen Pico-chan Toujou!! | 0.547047      |
| Grendizer Giga                                                 | 0.000000      |
| Pink Mizu Dorobou Ame Dorobou                                  | 0.000000      |
| En En Nikoli                                                   | 0.000000      |


<br>

### Result

The `anime_recommendations` function is created to find anime recommendations using the previously defined similarity. This function works by taking anime with the highest similarity from the existing indices.

Next is to find recommendations similar to the anime "Naruto":

| anime_id | name   | genre                                              | type | episodes | rating | members |
|----------|--------|----------------------------------------------------|------|----------|--------|---------|
| 20       | Naruto | Action, Comedy, Martial Arts, Shounen, Super Power | TV   | 220      | 7.81   | 683297  |

Here are the top 5 recommendations:

| name                                                    | genre                                              |
|---------------------------------------------------------|----------------------------------------------------|
| Naruto: Shippuuden Movie 4 - The Lost Tower             | Action, Comedy, Martial Arts, Shounen, Super Power |
| Naruto Shippuuden: Sunny Side Battle                    | Action, Comedy, Martial Arts, Shounen, Super Power |
| Boruto: Naruto the Movie - Naruto ga Hokage ni Natta Hi | Action, Comedy, Martial Arts, Shounen, Super Power |
| Naruto x UT                                             | Action, Comedy, Martial Arts, Shounen, Super Power |
| Naruto: Shippuuden                                      | Action, Comedy, Martial Arts, Shounen, Super Power |

The system has successfully recommended the top 5 anime that are similar to Naruto, including some movies and series from Naruto itself. So, if a user likes Naruto, the system can recommend other series or movies from Naruto.

## Evaluation
The evaluation of the system using the recommender system precision in finding recommendations from the anime "Naruto" is 5/5 or 100%.

![Recommender System Precision](https://user-images.githubusercontent.com/107544829/190915653-cf57d3de-db41-455c-b060-b4dd6630157b.png)
