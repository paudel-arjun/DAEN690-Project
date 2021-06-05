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
tidymodels_prefer()
```


```r
dt <- read_csv(here("Data/runway_incursion_narrative.csv")) %>% 
  clean_names()
```

```
## 
## -- Column specification --------------------------------------------------------
## cols(
##   `LOC ID` = col_character(),
##   NARRATIVE = col_character()
## )
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
  count(word) %>% 
  slice_max(n, n=50)

bigrams <- dt %>% 
  unnest_tokens(word, narrative, token = "ngrams", n = 2)

bigrams_cnts <- bigrams %>% 
  count(word) %>% 
  slice_max(n, n = 50)
  
bigrams_cnts %>% 
  ggplot(aes(x=n, y = fct_reorder(word, n)))+
  geom_col()
```

![](DAEN_690_files/figure-html/unnamed-chunk-1-1.png)<!-- -->

