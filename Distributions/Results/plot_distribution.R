library(ggplot2)
library(dplyr)
library(tidyr)
library(ggpubr)

plotting.path <- "D:/Angle/Code/Distributions/Results/"

df1 <- read.csv(paste(plotting.path,'power_model1.csv',sep=""),sep = '\t') %>% as.data.frame()
df2 <- read.csv(paste(plotting.path,'power_model2.csv',sep=""), sep = '\t') %>% as.data.frame()
df3 <- read.csv(paste(plotting.path,'power_model3.csv',sep=""), sep = '\t') %>% as.data.frame()

### model1 ####
p1 <- ggplot(df1, aes(x=delta, y=power, color=method, group=method)) + 
    geom_point(aes(shape=method)) + geom_line(aes(linetype=method), size=1) 

p2 <- ggplot(df2, aes(x=delta, y=power, color=method, group=method)) + 
  geom_point(aes(shape=method)) + geom_line(aes(linetype=method), size=1) 

p3 <- ggplot(df3, aes(x=delta, y=power, color=method, group=method)) + 
  geom_point(aes(shape=method)) + geom_line(aes(linetype=method), size=1) 

ggarrange(p1, p2, p3, nrow = 1, ncol = 3, common.legend = TRUE, legend = 'right')
