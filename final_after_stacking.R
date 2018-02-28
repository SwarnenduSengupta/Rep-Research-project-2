library(randomForest)
e<-0
for(i in seq(1,20,2))
{
  print(i)
  e<-e+predict_lr_acc_final[i]$predict_lr_acc
}
e<-e/10
f<-0
for(i in seq(1,20,2))
{
  print(i)
  f<-f+predict_lr_auc_final[i]$predict_lr_auc
}
f<-f/10
g<-0
for(i in seq(1,20,2))
{
  print(i)
  g<-g+predict_lr_conf_final[i]$predict_lr_conf
}
g<-g/10
h<-0
for(i in seq(2,30,3))
{
  print(i)
  h<-h+pred_ada_final[i]$'probs.1'
}
h<-h/10
j<-0
for(i in seq(1,30,3))
{
  print(i)
  j<-j+pred_rpart_final[i]$'0'
}
j<-j/10
k<-0
for(i in seq(1,10))
{
  print(i)
  k<-k+as.numeric(pred_rf_final[i]$pred_rf)
}
k<-k/10
l<-0
for(i in seq(1,10))
{
  print(i)
  l<-l+as.numeric(pred_knn_23_final_class[,i])
}
l<-(l/10)-1
l<-ifelse(l>0.5,1,0)
m<-0
for(i in seq(1,10))
{
  print(i)
  m<-m+as.numeric(pred_knn_23_final_prob[,i])
}
m<-m/10
m<-ifelse(l==0,m,1-m)

n<-0
for(i in seq(1,10))
{
  print(i)
  n<-n+as.numeric(pred_knn_36_final_class[,i])
}
n<-(l/10)-1
n<-ifelse(l>0.5,1,0)
p<-0
for(i in seq(1,10))
{
  print(i)
  p<-p+as.numeric(pred_knn_36_final_prob[,i])
}
p<-n/10
p<-ifelse(l==0,n,1-n)

final_stack_prob<-data.frame(e,f,g,h,j,k,m,p,as.factor(dataset$OUTPUT))

#class---------------
e1<-0
for(i in seq(2,20,2))
{
  print(i)
  e1<-e1+predict_lr_acc_final[i]$predicted
}
e1<-e1/10
e1<-ifelse(e1>0.5,1,0)
f1<-0
for(i in seq(2,20,2))
{
  print(i)
  f1<-f1+predict_lr_auc_final[i]$predicted
}
f1<-f1/10
f1<-ifelse(f1>0.5,1,0)
g1<-0
for(i in seq(2,20,2))
{
  print(i)
  g1<-g1+predict_lr_conf_final[i]$predicted
}
g1<-g1/10
g1<-ifelse(g1>0.5,1,0)
h1<-0
for(i in seq(1,30,3))
{
  print(i)
  h1<-h1+as.numeric(pred_ada_final[i]$class)
}
h1<-h1/20
h1<-ifelse(h1>0.75,1,0)
j1<-0
for(i in seq(3,30,3))
{
  print(i)
  j1<-j1+pred_rpart_final[i]$class
}
j1<-j1/10
j1<-ifelse(j1>0.2,1,0)
k1<-0
for(i in seq(1,10))
{
  print(i)
  k1<-k1+as.numeric(pred_rf_final[i]$pred_rf)
}
k1<-k1/10
k1<-ifelse(k1>=1.5,1,0)
#l,n
final_stack_class<-data.frame(e1,f1,g1,h1,j1,k1,l,n,as.factor(dataset$OUTPUT))
names(final_stack_prob)[9]<-"OUTPUT"
names(final_stack_class)[9]<-"OUTPUT"
oob_rf_class<-matrix()
acc_rf_class<-matrix()
auc_rf_class<-matrix()
pred_rf_final_class<-c()
oob_rf_prob<-matrix()
acc_rf_prob<-matrix()
auc_rf_prob<-matrix()
pred_rf_final_prob<-c()
Strt<-Sys.time()
for(i in seq(1,20))
{
  library(ModelMetrics)
#  output_matrix<-data.frame(t1=numeric())
  index <- sample(1:nrow(final_stack_class), nrow(final_stack_class))
  ind<-index[1:70000]
  train_class<-final_stack_class[ind,]
  train_prob<-final_stack_prob[ind,]
  #rf ----------
  rf_class<-randomForest(OUTPUT~.,data = train_class)
  rf_prob<-randomForest(OUTPUT~.,data = train_prob)

  pred_rf_class<-(predict(rf_class,dataset))
  pred_rf_class<-as.data.frame(pred_rf_class)
  pred_rf_final_class<-c(pred_rf_final_class,pred_rf_class)
  oob_rf_class[i]<-as.numeric(rf_class$err.rate[500,1])
  auc_rf_class[i]<-auc(dataset$OUTPUT,pred_rf_class$pred_rf_class)
  acc_rf_class[i]<-mean(dataset$OUTPUT==pred_rf_class$pred_rf_class)
  
  pred_rf_prob<-(predict(rf_prob,dataset))
  pred_rf_prob<-as.data.frame(pred_rf_prob)
  pred_rf_final_prob<-c(pred_rf_final_prob,pred_rf_prob)
  oob_rf_prob[i]<-as.numeric(rf_prob$err.rate[500,1])
  auc_rf_prob[i]<-auc(dataset$OUTPUT,pred_rf_prob$pred_rf_prob)
  acc_rf_prob[i]<-mean(dataset$OUTPUT==pred_rf_prob$pred_rf_prob)
  
  print("rf")
  
}
Stp<-Sys.time()