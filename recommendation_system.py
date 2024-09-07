
#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# loading dataset
data = pd.read_csv(r"C:\Users\rohan\OneDrive\Desktop\dissertation\amazon_co-ecommerce_sample.csv")
data.head()


# Display the first few rows of the dataset
print(data.head())

# Summary statistics
print(data.describe())

# Check for missing values
missing_values = data.isnull().sum()
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values.index, y=missing_values.values)
plt.title('Missing Values in Dataset')
plt.xticks(rotation= 90)
plt.xlabel('Features')
plt.ylabel('Count of Missing Values')
plt.show()



# Histogram of product prices
plt.figure(figsize=(15, 10))
sns.histplot(data['price'], bins=20, kde=True, color='red')
plt.title('Distribution of Product Prices', fontsize=16)
plt.xlabel('Price (in GBP)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Limit the y-axis (frequency) to a maximum value of 125
plt.ylim(0, 125)

# Define the desired tick locations and labels for the x-axis
x_ticks = [0, 500, 1000, 1500, 2000, 2500]  # Adjust these values as needed
x_tick_labels = ['£0', '£500', '£1000', '£1500', '£2000', '£2500']  # Corresponding labels

# Set the x-axis ticks and labels
plt.xticks(x_ticks, x_tick_labels, fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# Violin plot of review ratings
plt.figure(figsize=(10, 6))
data['average_review_rating'] = data['average_review_rating'].str.extract(r'([\d.]+)').astype(float)
sns.violinplot(x='average_review_rating', data=data, color='skyblue', inner='box')
plt.title('Distribution of Average Review Ratings', fontsize=16)
plt.xlabel('Average Review Rating', fontsize=12)
plt.xticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Countplot of product categories
plt.figure(figsize=(14, 10))  # Increase the figure size
sns.countplot(y='amazon_category_and_sub_category', data=data, order=data['amazon_category_and_sub_category'].value_counts().index)
plt.title('Count of Products in Each Category', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Category', fontsize=12)

# Rotate y-labels and adjust spacing
plt.xticks(fontsize=10)
plt.yticks(rotation=45, fontsize=10)
plt.subplots_adjust(left=0.25)  # Adjust left margin to prevent overlapping y-labels

plt.show()

"""Relationships Between Variables:

To build effective recommendation systems, we need to understand how variables relate to each other. Here, we'll explore relationships between price, ratings, and the number of reviews.
"""

# Pairplot for numerical variables
sns.pairplot(data[['price', 'number_of_reviews', 'average_review_rating']])
plt.suptitle('Pairplot of Numerical Variables', y=1.02)
plt.show()

# Scatterplot of price vs. average_review_rating
plt.figure(figsize=(8, 6))
sns.scatterplot(x='price', y='average_review_rating', data=data)
plt.title('Price vs. Average Review Rating')
plt.xlabel('Price(in GBP)')
plt.ylabel('Average Review Rating')
# Define the desired tick locations and labels for the x-axis
x_ticks = [0, 500, 1000, 1500, 2000, 2500]  # Adjust these values as needed
x_tick_labels = ['£0', '£500', '£1000', '£1500', '£2000', '£2500']  # Corresponding labels

# Set the x-axis ticks and labels
plt.xticks(x_ticks, x_tick_labels, fontsize=10)
plt.show()

# Scatterplot of number_of_reviews vs. average_review_rating
plt.figure(figsize=(8, 6))
sns.scatterplot(x='number_of_reviews', y='average_review_rating', data=data)
plt.title('Number of Reviews vs. Average Review Rating')
plt.xlabel('Number of Reviews')
plt.ylabel('Average Review Rating')
# Define the desired tick locations and labels for the x-axis
x_ticks = [0, 25, 50,75, 100,125, 150,175, 200]  # Adjust these values as needed
x_tick_labels = ['0', '25', '50', '75','100','125', '150','175', '200']  # Corresponding labels

# Set the x-axis ticks and labels
plt.xticks(x_ticks, x_tick_labels, fontsize=10)
plt.show()

"""**Preprocessing**"""

data.head()

# I thought of removing these columns not necessary for hybrid recommendation system
data.drop(columns=['number_available_in_stock','number_of_answered_questions','product_description','customer_questions_and_answers', 'customer_reviews'],inplace=True)
data.head()

# checking again number of null values
data.isna().sum()

# again thought of removing these columns for not using in hybrid recommendation system
data.drop(columns = ['customers_who_bought_this_item_also_bought','items_customers_buy_after_viewing_this_item','sellers'])
data.head()

# again thought of removing these columns for not using in my recommendation
data.drop(columns= ['customers_who_bought_this_item_also_bought','product_name','sellers'],inplace=True)
data.head()

# this one also thought of to be dropped
del data['items_customers_buy_after_viewing_this_item']
data.head()

# checking number of null values again in these columns
data.isna().sum()

# filtering the numerical and categorical columns
num_cols = ['number_of_reviews','average_review_rating']
cat_cols = list(set(data.columns)-set(num_cols))
cat_cols

# replacing null values by 0 in the categorical columns
for col in num_cols:
    data[col].fillna(0,inplace=True)
data.isna().sum()

# replacing null values by empty strings in categorical columns
for col in cat_cols:
    data[col].fillna('',inplace=True)
data.isna().sum()

# let's have a look again on our dataset
data.head()

# although we have filtered out number_of_reviews and average_review_rating as numerical columns
#but those are still object type values
data.dtypes

# in our dataset the number_of_review columns need to be processed as there are ',' punctuation in this column
for ind in data.index:
#     data['number_of_reviews'][ind] = int(data['number_of_reviews'][ind].replace(',',''))
    try:
        data['number_of_reviews'][ind] = int(data['number_of_reviews'][ind].replace(',',''))
    except:
        pass
data.dtypes

# let's have a look again on our dataset
data.head()

# now let's process our average_review_rating column
for ind in data.index:
#     df['average_review_rating'][ind] = float(df['average_review_rating'][ind].split(' ')[0])
    try:
        data['average_review_rating'][ind] = float(data['average_review_rating'][ind].split(' ')[0])
    except:
        pass
data.dtypes

# let's have another look on our dataset
data.head()

#changin the column type
for col in num_cols:
    data[col] = data[col].astype(float)
data.dtypes

# let's see a short description of our dataset to determine the weighted rating parameters
data.describe(include='all')

# Commented out IPython magic to ensure Python compatibility.
# here we can see number of reviews is highly connected with average review ratings
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set()
plt.figure(figsize=(8,5))

plot = sns.scatterplot(x=data["number_of_reviews"],
                       y=data["average_review_rating"])

# let's determine weighted rating parameters
data['weighted_rating'] = 0
minimum_num_of_ratings = 9
mean_vote = 4.6
data.head()

# calculating the weighted_rating
for ind in data.index:
#     df['weighted_rating'][ind] = ((df['average_review_rating'][ind] * df['number_of_reviews'][ind])+(mean_vote*minimum_num_of_ratings))/(df['average_review_rating'][ind] + df['number_of_reviews'][ind])
#     print(df['weighted_rating'][ind])
    try:
         data['weighted_rating'][ind] = ((data['average_review_rating'][ind] * data['number_of_reviews'][ind])+(mean_vote*minimum_num_of_ratings))/(data['average_review_rating'][ind] + data['number_of_reviews'][ind])
    except:
        data['weighted_rating'][ind] = 1
data.head()

# let's look on the description of our dataset again
data.describe()

# as we have calculated weighted rating so these two columns are no longer needed
data.drop(columns = ['number_of_reviews','average_review_rating'],inplace=True)
data.head()

# let's check up the category to determine what to process in the string
data['amazon_category_and_sub_category'][0]

# I have thought of deleting the description too
del data['description']
data.head()

# so now the shape is 10000,5 from 10000, 17
data.shape

# processing the category column strings
for ind in data.index:
    data['amazon_category_and_sub_category'][ind] = data['amazon_category_and_sub_category'][ind].replace('>','')

# looks like all good in the category column
data['amazon_category_and_sub_category'][0]

# now let's have a look on our product information to see what we need to process
data['product_information'][0]

# I have thought of processing all the strings together. So joined all of them to product details column and dropped the used
# columns
data['product_details'] = data['product_information'] + ' ' + data['manufacturer'] + ' ' + data['amazon_category_and_sub_category']
data.drop(columns = ['product_information', 'manufacturer', 'amazon_category_and_sub_category'],inplace=True)
data.head()

data['product_details'] = data['product_details'].str.lower()
data.head()

# so our ultimate dataset shape is 10000,3 from 10000, 17!
data.shape

# importing natural language processing libraries to process strings
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
nltk.download('punkt')

# removing the stop words from the dataset
stop_words = set(stopwords.words('english'))
for ind in data.index:
    example_sent = data['product_details'][ind]
    word_tokens = word_tokenize(example_sent)

    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    data['product_details'][ind] = ' '.join(filtered_sentence)
data.head()

# define punctuation
# removing the punctuations in the strings
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
for ind in data.index:

    my_str = data['product_details'][ind]
    # remove punctuation from the string
    no_punct = ""
    for char in my_str:
       if char not in punctuations:
           no_punct = no_punct + char

    data['product_details'][ind] = no_punct
data.head()

# looks all good
data['product_details'][0]

# so we are going to make a vector of words to make word to word n_gram and calculate cosine similarities
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# making the matrix
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(data['product_details'])

# calculating the cosine_similarities
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# here we are multipling the weighted rating of every of product with their cosine similarities
# so that both of the recommendation have an impact
for x in range(len(cosine_similarities)):
    cosine_similarities[x] = cosine_similarities[x]*data['weighted_rating']/9*100

# calculating the results for recommending
results = {}

for idx, row in data.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], data['uniq_id'][i]) for i in similar_indices]

    results[row['uniq_id']] = similar_items[1:]

print('done!')

# defining a function to recommend items
def item(id):
    return data.loc[data['uniq_id'] == id]['product_details'].tolist()[0].split(' - ')[0]


# Replace 'inf' values with NaN in the 'weighted_review' column
data['weighted_rating'] = data['weighted_rating'].replace([np.inf, -np.inf], np.nan)

# Drop rows with NaN values in the 'weighted_review' column
data = data.dropna(subset=['weighted_rating'])

# Now use this cleaned data to calculate actual_ratings
actual_ratings = data['weighted_rating'].tolist()

#  (truncate or pad as needed)
if len(actual_ratings) < 81:
    actual_ratings.extend([0.0] * (81 - len(actual_ratings)))
elif len(actual_ratings) > 81:
    actual_ratings = actual_ratings[:81]



# Modify the recommend function to return predicted ratings
def recommend_with_ratings(item_id, num):
    print("Recommending " + str(num) + " products similar to " + item(item_id)[:150] + "...")
    print("-------")
    recs = results[item_id][:num]
    recommendations = []
    
    for rec in recs:
        recommended_item_id = rec[1]
        predicted_rating = rec[0]
        if not np.isinf(predicted_rating):  # Check if predicted rating is not 'inf'
            recommendations.append((recommended_item_id, predicted_rating))
            print("Recommended: " + item(recommended_item_id)[:150] + " (predicted rating: " + str(round(predicted_rating, 2)) + ")")
    
    return recommendations

# Call the modified function to get recommendations with predicted ratings
recommendations_with_ratings = recommend_with_ratings(item_id='eac7efa5dbd3d667f26eb3d3ab504464', num=10000)

# Extract predicted ratings from recommendations
predicted_ratings = [rec[1] for rec in recommendations_with_ratings]

"""There were a total of 18 items in the dataset that did not receive any ratings. If we had exclusively relied on a rating-based recommendation system, these 18 items would have been excluded from the recommendation process. Therefore, our approach partially addresses the cold start problem by ensuring that even items without ratings have the opportunity to receive recommendations, contributing to a more comprehensive and effective recommendation system with favorable recommendation scores."""

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np



# Calculate MAE, RMSE, and MSE
mae = mean_absolute_error(actual_ratings, predicted_ratings)
rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
mse = mean_squared_error(actual_ratings, predicted_ratings)

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Squared Error (MSE):", mse)




# Define the MAE, MSE, and RMSE values
mae_value = 16.822
mse_value = 17.604
rmse_value = 309.930

# Create a bar plot with values
plt.figure(figsize=(10, 6))
sns.barplot(x=['MAE', 'MSE', 'RMSE'], y=[mae_value, mse_value, rmse_value], palette='viridis')
plt.title('Evaluation Metrics')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.ylim(0, 350)  # Adjust the y-axis limit as needed

# Add the values on top of the bars
for i, v in enumerate([mae_value, mse_value, rmse_value]):
    plt.text(i, v + 10, str(round(v, 2)), ha='center', va='bottom', fontsize=12, color='black')

# Show the plot
plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the threshold
threshold = 6.0

# Convert actual_ratings and predicted_ratings to NumPy arrays
actual_ratings = np.array(actual_ratings)
predicted_ratings = np.array(predicted_ratings)

# Create binary labels based on the threshold
actual_labels = np.where(actual_ratings >= threshold, 1, 0)
predicted_labels = np.where(predicted_ratings >= threshold, 1, 0)

# Calculate classification metrics
accuracy = accuracy_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels)
recall = recall_score(actual_labels, predicted_labels)
f1 = f1_score(actual_labels, predicted_labels)

# Print the classification metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

import matplotlib.pyplot as plt
import seaborn as sns

# Define the metrics and their values
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [0.888, 0.888, 1.0, 0.941]

# Set a modern style using seaborn
sns.set(style="whitegrid")

# Create a bar plot
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x=metrics, y=values, palette="viridis")
plt.xlabel("Metrics")
plt.ylabel("Values")
plt.title("Classification Metrics", y=1.1)
plt.ylim(0, 1)  # Set the y-axis limits to 0-1 for readability

# Add values on top of the bars
for i, v in enumerate(values):
    barplot.text(i, v + 0.02, str(round(v, 3)), ha='center', va='bottom', fontsize=12, color='black')

# Show the plot
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the threshold
threshold = 6.0

# Convert actual_ratings and predicted_ratings to NumPy arrays
actual_ratings = np.array(actual_ratings)
predicted_ratings = np.array(predicted_ratings)

# Create binary labels based on the threshold
actual_labels = np.where(actual_ratings >= threshold, 1, 0)
predicted_labels = np.where(predicted_ratings >= threshold, 1, 0)

# Create a confusion matrix
conf_matrix = confusion_matrix(actual_labels, predicted_labels)

# Define class labels
class_labels = ['Not Recommended', 'Recommended']

# Set a modern style using seaborn
sns.set(style="whitegrid")

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Create a heatmap for the confusion matrix with darker colors
heatmap = sns.heatmap(conf_matrix, annot=False, cmap="Blues", cbar=True,
                      xticklabels=class_labels, yticklabels=class_labels, ax=ax1)

# Set font size for axis labels
ax1.tick_params(axis='both', labelsize=12)

ax1.set_xlabel('Predicted', fontsize=14)
ax1.set_ylabel('Actual', fontsize=14)
ax1.set_title('Confusion Matrix', fontsize=16, pad=20)  # Add space for the title

# Place text annotations outside the matrix
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        ax1.text(j + 0.5, i + 0.5, str(conf_matrix[i, j]), ha='center', va='center', color='black')

# Calculate classification metrics
accuracy = accuracy_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels)
recall = recall_score(actual_labels, predicted_labels)
f1 = f1_score(actual_labels, predicted_labels)

# Add a separate area for displaying metrics
metrics_text = f"Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1 Score: {f1:.3f}"
ax2.text(0.2, 0.2, metrics_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Remove ticks and labels from the second subplot
ax2.axis('off')

# Show the plot
plt.show()
