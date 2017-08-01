library("tidyverse")
train <- read_csv("./raw_data/train.csv")
test <- read_csv("./raw_data/test.csv")

# Update data types

train <- train %>% mutate(
  Survived = factor(Survived),
  Pclass = factor(Pclass),
  Embarked = factor(Embarked),
  Sex = factor(Sex)
)

test <- test %>% mutate(
  Pclass = factor(Pclass),
  Embarked = factor(Embarked),
  Sex = factor(Sex)
)

# Datasets profiling 
library("Hmisc")
describe(train)
#Cabin, Embarked and Age have missing values. We need to impute/do something.

# Missing values
library("VIM")
aggr(train, prop = FALSE, combined = TRUE, numbers = TRUE, sortVars = TRUE, sortCombs = TRUE)
# Most cases have Cabin == NA

# Representation Survived vs important features (https://www.kaggle.com/headsortails/tidy-titarnic)

p_age = ggplot(train) +
  geom_freqpoly(mapping = aes(x = Age, color = Survived), binwidth = 1) +
  guides(fill=FALSE) +
  theme(legend.position = "none")

p_sex = ggplot(train, mapping = aes(x = Sex, fill = Survived)) +
  geom_bar(stat='count', position='fill') +
  labs(x = 'Sex') +
  scale_fill_discrete(name="Surv") +
  theme_few()

p_class = ggplot(train, mapping = aes(x = Pclass, fill = Survived, colour = Survived)) +
  geom_bar(stat='count', position='fill') +
  labs(x = 'Pclass') +
  theme(legend.position = "none")

p_emb = ggplot(train, aes(Embarked, fill = Survived)) +
  geom_bar(stat='count', position='fill') +
  labs(x = 'Embarked') +
  theme(legend.position = "none")

p_sib = ggplot(train, aes(SibSp, fill = Survived)) +
  geom_bar(stat='count', position='fill') +
  labs(x = 'SibSp') +
  theme(legend.position = "none")

p_par = ggplot(train, aes(Parch, fill = Survived)) +
  geom_bar(stat='count', position='fill') +
  labs(x = 'Parch') +
  theme(legend.position = "none")

p_fare = ggplot(train) +
  geom_freqpoly(mapping = aes(Fare, color = Survived), binwidth = 0.05) +
  scale_x_log10() +
  theme(legend.position = "none")


layout <- matrix(c(1,1,2,3,3,4,5,6,7),3,3,byrow=TRUE)
multiplot(p_age, p_sex, p_fare, p_class, p_emb, p_sib, p_par, layout=layout)

# Correlations between numeric variables
library("corrplot")
train %>%
  select(-PassengerId, -Name, -Cabin, -Ticket) %>%
  mutate(Sex = factor(case_when(.$Sex == 'male' ~ '0',
                         .$Sex == 'female' ~ '1',
                         TRUE ~ '0'
  ))
  ) %>%
  mutate(Sex = as.integer(Sex),
         Pclass = as.integer(Pclass),
         Survived = as.integer(Survived),
         Embarked = as.integer(Embarked)) %>%
  cor(use="complete.obs") %>%
  corrplot(type="lower", diag=FALSE)

# Impute NAs: Age, Embarked
combine  <- bind_rows(train, test) # bind training & test data
combine_lm <- combine %>% filter(!is.na(Age)) %>% select(Age,Pclass, Sex, SibSp, Parch, Fare) %>%
  mutate(Sex = if_else(Sex == "male", 1, 0)) %>% 
  mutate_all(funs(as.numeric))
# Another approach: Find closest to instances with NA and replace by a statistics of the neighbourhood
age.mod <- lm(Age ~ Pclass + Sex +
                SibSp + Parch + Fare, data = combine_lm)
combine$Age[is.na(combine$Age)] <- predict(age.mod, combine %>% mutate(Sex = if_else(Sex == "male", 1, 0)))[is.na(combine$Age)]
combine <- combine %>%
  mutate(Embarked = as.character(Embarked)) %>%
  mutate(Embarked = case_when(
    is.na(.$Embarked) ~ "C",
    TRUE ~ .$Embarked
  )) %>%
  mutate(Embarked = as.factor(Embarked))

med_fare_3 <- combine %>%
  filter(!is.na(Fare)) %>%
  group_by(Pclass) %>% 
  summarise(med_fare = median(Fare)) %>%
  filter(Pclass == 3) %>%
  .$med_fare

combine <- combine %>%
  mutate(Fare = case_when(
    is.na(.$Fare) ~ med_fare_3,
    TRUE ~ .$Fare
  ))


# New features (Engineered)
library("stringr")
combine <- mutate(combine,
                  fclass = factor(log10(Fare+1) %/% 1),
                  age_known = factor(!is.na(Age)),
                  cabin_known = factor(!is.na(Cabin)),
                  title_orig = factor(str_extract(Name, "[A-Z][a-z]*\\.")),
                  young = factor( if_else(Age<=30, 1, 0, missing = 0) | (title_orig %in% c('Master.','Miss.','Mlle.')) ),
                  child = Age<10,
                  family = SibSp + Parch,
                  alone = (SibSp == 0) & (Parch == 0),
                  large_family = (SibSp > 2) | (Parch > 3),
                  deck = if_else(is.na(Cabin),"U",str_sub(Cabin,1,1)),
                  ttype = str_sub(Ticket,1,1)
)
library("forcats")
combine <- mutate(combine, title = fct_lump(title_orig, n=4))

train_cleaned <- combine %>% filter(!is.na(Survived))
test_cleaned <- combine %>% filter(is.na(Survived))
