setwd("X:/Workspace")
mydata <- read.csv("sharktank.csv")
str(mydata)
View(mydata)
description <- as.character(mydata$description)
library(dplyr)
glimpse(description)
head(description)
library(tm)
#creating the text corpus
corpus <- SimpleCorpus(VectorSource(description))
View(corpus)

#Steps involved in cleaning the text corpus are
#1. Stripping any extra white space
corpus <- tm_map(corpus,stripWhitespace)

#2. Transforming everything to lowercase
corpus <- tm_map(corpus, content_transformer(tolower))

# 3. Removing numbers 
corpus <- tm_map(corpus, removeNumbers)

# 4. Removing punctuation
corpus <- tm_map(corpus, removePunctuation)

# 5. Removing stop words
corpus <- tm_map(corpus, removeWords, c(stopwords("english"),stopwords("smart"),
                                        "everytime","get","using","help","","designed", "products","can"
                                        ,"make", "makes", "company","one", "use", "back", "product","without"
                                        , "making","made","also", "just", "even","like"))

View(corpus)


stopwords("english")
stopwords("smart")

#Lemmatization our data so that the words go to root words
library(textstem)
corpus <- tm_map(corpus,lemmatize_words)
View(corpus)

#creating the Document text matrix
dtmforcloud <- DocumentTermMatrix(corpus)
DTM <- (DocumentTermMatrix(corpus))
View(DTM)
inspect(DTM)


sums <- as.data.frame(as.matrix(DTM))
View(sums)
colnames(sums)


#Preparing the data for word cloud
cloud<- as.data.frame(colSums(as.matrix(dtmforcloud)))
head(cloud)
library(tibble)
cloud <- rownames_to_column(cloud)
head(cloud)
colnames(cloud) <-c("words", "frequency")
cloud <- arrange(cloud, desc(frequency))
head(cloud)
head <- cloud[1:75,]

#Word cloud
library(wordcloud)
wordcloud(words=head$words, freq= head$frequency, min.freq=1,
          max.words=4000, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8,"Dark2"))

dataset = sums
str(dataset)
dataset = removeSparseTerms(DTM, 0.98)
View(dataset)
dataset=as.data.frame(as.matrix(dataset))
dataset$TF = mydata$deal
head(dataset$TF)
View(dataset)

#splitting data into training and testing. 
library(caTools)
set.seed(123)
split = sample.split(dataset, SplitRatio = 0.7)
training_set = subset(dataset, split == TRUE)
#training_set$TF <- as.factor(training_set$TF)
test_set = subset(dataset, split == FALSE)
tester <- test_set$TF
glimpse(tester)
newtest_set <- test_set[-70]
str(newtest_set)

#building the cart model. 
library(rpart)
library(caret)
modfit <- rpart(TF~.,method="class",data=training_set, 
                control=list(cp=0, xval=0 )) 
check <- predict(modfit, newtest_set, type="class")
(confusionCART <- table(check,tester))

#calculating the accuracy of the model. 
(accuracyCART <- sum(diag(confusionCART))/sum(confusionCART))

printcp(modfit)
tree <- modfit
tree <- prune(tree,cp=0.00,"cp")
library(rpart.plot)
rpart.plot(tree)

#building a logistic regression model
logfit <- glm(TF~., data=training_set, family="binomial")
logpred <- predict(logfit, newtest_set, response="prob")
(confLogit <- table(check,tester))
#Accuracy of Logistical model
(accuracyLogit <- mean(check==tester))

#bulding randomforest
library(randomForest)
randomfit <- randomForest(TF~., data= training_set)
randompred <- predict(randomfit, newtest_set)
#accuracy
(accuracyRandom <- mean(check==tester))

#VarImpPlot is a Dotchart of variable importance as measured by a Random Forest
varImpPlot(randomfit)

#step 2: 
#Adding the ratio of askefor/Valuation
(ratio <- mydata$askedFor/mydata$valuation)
glimpse(dataset)
newdataset <- dataset
newdataset$ratio <- ratio
glimpse(newdataset)


#newdata split
newsplit = sample.split(newdataset, SplitRatio = 0.8)
newtraining_set = subset(newdataset, newsplit == TRUE)

#training_set$TF <- as.factor(training_set$TF)
newtest_set = subset(newdataset, newsplit == FALSE)
newtester <- newtest_set$TF
glimpse(newtester)
yeltest_set <- newtest_set[-70]
glimpse(yeltest_set)

#building the cart model with the newtraining dataset
newmodfit <- rpart(TF~.,method="class",data=newtraining_set, 
                control=list(cp=0.0, xval=10 )) 
newcheck <- predict(newmodfit, yeltest_set, type="class")
(confusionCART <- table(newcheck,newtester))
#calculating the accuracy of the model
(newaccuracyCART <- mean(newcheck==newtester))
newmodfit <- prune(newmodfit,cp=0.0,"cp")
rpart.plot(newmodfit)

#Bulding the Logistic model we have
newlogfit <- glm(TF~., data=newtraining_set, family="binomial")
newlogpred <- predict(newlogfit, yeltest_set, response="prob")
(confLogit <- table(newcheck,newtester>0.70))
#Accuracy of Logistical model
(newaccuracyLogit <- mean(newcheck==(newtester>0.70)))


#Building the RandomForest model we have. 
newrandomfit <- randomForest(TF~., data= newtraining_set)
newrandompred <- predict(newrandomfit, yeltest_set)
#accuracy
(newaccuracyRandom <- mean(newcheck==newtester))
#VarImpPlot is a Dotchart of variable importance as measured by a Random Forest
varImpPlot(newrandomfit)


#Comparision of accuracies across model
(accuracyCART <- sum(diag(confusionCART))/sum(confusionCART))
(accuracyLogit <- mean(check==tester))
(accuracyRandom <- mean(check==tester))

#Comparision of the new dataset accuracies
(newaccuracyCART <- mean(newcheck==newtester))
(newaccuracyLogit <- mean(newcheck==newtester))
(newaccuracyRandom <- mean(newcheck==newtester))
