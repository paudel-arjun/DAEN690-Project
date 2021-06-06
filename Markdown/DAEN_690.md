---
title: "Text_analysis"
author: "Arjun Paudel"
date: "6/5/2021"
output: 
  html_document: 
    keep_md: yes
editor_options: 
  chunk_output_type: console
---




```r
library(tidyverse)
library(tidytext)
library(textrecipes)
library(widyr)
library(tidymodels)
library(here)
library(janitor)
library(hardhat)
library(themis)
tidymodels_prefer()
```


```r
dt <- read_csv(here("Data/runway_incursion_narrative.csv")) %>% 
  clean_names()
dt <- dt %>% 
  filter(cat_rank %in% c("C", "D")) %>% 
  mutate(cat_rank = factor(cat_rank)) 
```


```r
tokens <- dt %>% 
  unnest_tokens(word, narrative) %>% 
  anti_join(get_stopwords())
```

```
## Joining, by = "word"
```

```r
token_cnts <- tokens %>% 
  select(-loc_id) %>% 
  count(cat_rank, word) %>% 
  group_by(cat_rank) %>% 
  slice_max(n, n=20, with_ties = FALSE)

token_cnts %>% 
  ggplot(aes(x = n, 
             y = reorder_within(word, n, cat_rank),
             fill = cat_rank)) +
  geom_col() +
  facet_wrap(vars(cat_rank), scales = "free")+
  scale_y_reordered()
```

![](DAEN_690_files/figure-html/eda-1.png)<!-- -->

```r
bigrams <- dt %>% 
  unnest_tokens(word, narrative, token = "ngrams", n = 2)

bigrams_cnts <- bigrams %>% 
  count(word) %>% 
  slice_max(n, n = 20)
  
bigrams_cnts %>% 
  ggplot(aes(x=n, y = fct_reorder(word, n)))+
  geom_col()
```

![](DAEN_690_files/figure-html/eda-2.png)<!-- -->


```r
initialsplit <- initial_split(dt, strata = cat_rank )
dt_train <- training(initialsplit)
dt_test <- testing(initialsplit)
folds <- vfold_cv(dt_train, strata = cat_rank)
```


```r
#use_glmnet(cat_rank ~ ., data = dt_train)
```


```r
sparse_bp <- default_recipe_blueprint(composition = "dgCMatrix")

glmnet_recipe <- 
  recipe(cat_rank ~ ., data = dt_train) %>% 
  update_role(loc_id, new_role = "id") %>% 
  step_tokenize(narrative) %>% 
  step_stopwords(narrative) %>% 
  step_tokenfilter(narrative, max_times = 1000, 
                   min_times = 100, max_tokens = 5000) %>% 
  step_tfidf(narrative) %>% 
  step_smote(cat_rank)
  
glmnet_spec <- 
  logistic_reg(penalty = tune(), mixture = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("glmnet") 

glmnet_workflow <- 
  workflow() %>% 
  add_recipe(glmnet_recipe, blueprint = sparse_bp) %>% 
  add_model(glmnet_spec) 

glmnet_grid <- tidyr::crossing(penalty = 10^seq(-3, -2, length.out = 20), mixture = c(0.05, 
    0.2, 0.4, 0.6, 0.8, 1)) 

doParallel::registerDoParallel()

glmnet_tune <- 
  tune_grid(glmnet_workflow, 
            resamples = folds, grid = glmnet_grid,   
            metrics = metric_set(accuracy, recall, precision, roc_auc),
            control = control_grid(save_pred = TRUE)) 
```


```r
autoplot(glmnet_tune)
```

![](DAEN_690_files/figure-html/metrics-1.png)<!-- -->

```r
glmnet_tune %>% show_best("accuracy")
```

```
## # A tibble: 5 x 8
##   penalty mixture .metric  .estimator  mean     n std_err .config               
##     <dbl>   <dbl> <chr>    <chr>      <dbl> <int>   <dbl> <chr>                 
## 1 0.00886     0.6 accuracy binary     0.784    10 0.00390 Preprocessor1_Model079
## 2 0.00546     1   accuracy binary     0.783    10 0.00417 Preprocessor1_Model115
## 3 0.00616     0.8 accuracy binary     0.783    10 0.00360 Preprocessor1_Model096
## 4 0.00483     1   accuracy binary     0.783    10 0.00375 Preprocessor1_Model114
## 5 0.00785     0.6 accuracy binary     0.783    10 0.00382 Preprocessor1_Model078
```

```r
glmnet_tune %>% show_best("roc_auc")
```

```
## # A tibble: 5 x 8
##   penalty mixture .metric .estimator  mean     n std_err .config               
##     <dbl>   <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>                 
## 1 0.00695     0.6 roc_auc binary     0.847    10 0.00415 Preprocessor1_Model077
## 2 0.01        0.4 roc_auc binary     0.847    10 0.00418 Preprocessor1_Model060
## 3 0.00886     0.4 roc_auc binary     0.847    10 0.00421 Preprocessor1_Model059
## 4 0.00428     1   roc_auc binary     0.847    10 0.00412 Preprocessor1_Model113
## 5 0.00616     0.6 roc_auc binary     0.847    10 0.00418 Preprocessor1_Model076
```

```r
best_param <- glmnet_tune %>% select_best("roc_auc")
```


```r
glmnet_workflow_final <- glmnet_workflow %>% 
  finalize_workflow(parameters = best_param )

final_fit <- glmnet_workflow_final %>%
  last_fit(split = initialsplit)
```


```r
final_fit %>% collect_metrics()
```

```
## # A tibble: 2 x 4
##   .metric  .estimator .estimate .config             
##   <chr>    <chr>          <dbl> <chr>               
## 1 accuracy binary         0.794 Preprocessor1_Model1
## 2 roc_auc  binary         0.862 Preprocessor1_Model1
```

```r
final_fit %>% collect_predictions() %>% 
  conf_mat(truth = cat_rank, estimate = .pred_class)
```

```
##           Truth
## Prediction    C    D
##          C 1254  387
##          D  503 2175
```

