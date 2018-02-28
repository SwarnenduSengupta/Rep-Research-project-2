  rm(list=ls())
  Start<-Sys.time()
  options(menu.graphics=FALSE)
  detach("package:nnet")
  #install.packages("nnet")
  setwd("/home/swarnendu/Documents/")
  dataset<-read.csv("input_2.csv")
  dataset<-dataset[-which(dataset$state1==""),]
  dataset$Country<-NULL
  dataset$Act_code<-NULL
  dataset$custAttr2<-NULL
  dataset$custAttr1<-NULL
  dataset$total<-NULL
  dataset$state1<-as.character(dataset$state1)
  dataset$state1<-as.factor(dataset$state1)
  dataset$OUTPUT<-as.factor(dataset$OUTPUT)
  state<-as.data.frame(unique(sort(dataset$state1)))
  state$col_name<-seq(1:dim(state)[1])
  names(state)[1]<-"state1"
  dataset<-merge(dataset,state,by="state1")
  dataset$state1<-NULL
  
  #libraries -------------
  library(randomForest)
  library(caret)
  library(nnet)
  library(e1071)
  library(ada)
  library(class)
  library(C50)
  library(rpart)
  #library(NeuralNetTools)
  #library(neuralnet)
  
  #actual work----------------
  #defining all the dataframes-----------
  conf_rf<-matrix()
  acc_rf<-matrix()
  auc_rf<-matrix()
  
  acc_knn36_final<-matrix()
  auc_knn36_final<-matrix()
  conf_knn36_final<-matrix()
  acc_knn23_final<-matrix()
  auc_knn23_final<-matrix()
  conf_knn23_final<-matrix()
  
  auc_lr_acc<-matrix()
  acc_lr_acc<-matrix()
  conf_lr_acc<-matrix()
  auc_lr_auc<-matrix()
  acc_lr_auc<-matrix()
  conf_lr_auc<-matrix()
  auc_lr_conf<-matrix()
  acc_lr_conf<-matrix()
  conf_lr_conf<-matrix()
  
  acc_nn_a<-matrix()
  auc_nn_a<-matrix()
  conf_nn_a<-matrix()
  acc_nn_conf<-matrix()
  auc_nn_conf<-matrix()
  conf_nn_conf<-matrix()
  
  conf_svm<-matrix()
  acc_SVM<-matrix()
  auc_SVM<-matrix()
  
  conf_c50<-matrix()
  acc_c50<-matrix()
  auc_c50<-matrix()
  conf_c50_win<-matrix()
  acc_c50_win<-matrix()
  auc_c50_win<-matrix()
  conf_rpart<-matrix()
  acc_rpart<-matrix()
  auc_rpart<-matrix()
  
  conf_ada<-matrix()
  acc_ada<-matrix()
  auc_ada<-matrix()
  
  output_matrix<-data.frame(t1=numeric())
  output_matrix_class<-data.frame(t1=numeric())
  total_corr<-data.frame(t1=numeric())
  total_corr_class<-data.frame(t1=numeric())
  pred_rf_final<-c()
  predict_lr_acc_final<-c()
  predict_lr_auc_final<-c()
  predict_lr_conf_final<-c()
  predict_nn_auc_acc_final<-data.frame(t1=numeric(),t2=numeric())
  predict_nn_conf_final<-data.frame(t1=numeric(),t2=numeric())
  pred_svm_final_class<-data.frame(t1=numeric())
  pred_svm_final_prob<-data.frame(t1=numeric(),t2=numeric())
  pred_ada_final<-c()
  pred_c50_final<-c()
  pred_c50_win_final<-c()
  pred_rpart_final<-c()
  pred_knn_36_final_prob<-data.frame(t1=numeric(),t2=numeric())
  pred_knn_36_final_class<-data.frame(t1=numeric())
  pred_knn_23_final_prob<-data.frame(t1=numeric(),t2=numeric())
  pred_knn_23_final_class<-data.frame(t1=numeric())
  for( i in seq(1,10))
  {
    library(ModelMetrics)
    output_matrix<-data.frame(t1=numeric())
    index <- sample(1:nrow(dataset), nrow(dataset))
    ind<-index[1:70000]
    train<-dataset[ind,]
      #rf ----------
    rf<-randomForest(OUTPUT~.,data = train,ntree=200)
    pred_rf<-(predict(rf,dataset,type = "prob"))
    pred_rf<-as.data.frame(pred_rf)
    pred_rf$pred_rf<-ifelse(pred_rf$`0`>pred_rf$`1`,0,1)
    pred_rf_final<-c(pred_rf_final,pred_rf)
    cnf_rf<-ifelse(pred_rf$`0`>pred_rf$`1`,pred_rf$`1`,pred_rf$`0`)
    conf_rf[i]<-mean(cnf_rf)
    auc_rf[i]<-auc(dataset$OUTPUT,pred_rf$pred_rf)
    acc_rf[i]<-mean(dataset$OUTPUT==pred_rf$pred_rf)
    
    print("rf")
  
    #LR----------------
    #acc
    model_lr<-glm(OUTPUT~.,data = train,family = binomial())
    predict_lr_acc<-predict(model_lr,dataset,type = "response")
    predict_lr_acc<-as.data.frame(predict_lr_acc)
    predict_lr_acc$predicted<-ifelse(predict_lr_acc$predict_lr_acc>0.1565,1,0)
    predict_lr_acc_final<-c(predict_lr_acc_final,predict_lr_acc)
    auc_lr_acc[i]<-auc(dataset$OUTPUT,predicted = predict_lr_acc$predicted)
    acc_lr_acc[i]<-mean(dataset$OUTPUT==predict_lr_acc$predicted)
    conf_lr_acc[i]<-mean(predict_lr_acc$predict_lr_acc)
    #auc
    predict_lr_auc<-predict(model_lr,dataset,type = "response")
    predict_lr_auc<-as.data.frame(predict_lr_auc)
    predict_lr_auc$predicted<-ifelse(predict_lr_auc$predict_lr_auc>0.0341,1,0)
    predict_lr_auc_final<-c(predict_lr_auc_final,predict_lr_auc)
    auc_lr_auc[i]<-auc(dataset$OUTPUT,predicted = predict_lr_auc$predicted)
    acc_lr_auc[i]<-mean(dataset$OUTPUT==predict_lr_auc$predicted)
    conf_lr_auc[i]<-mean(predict_lr_auc$predict_lr_auc)
    #conf
    predict_lr_conf<-predict(model_lr,dataset,type = "response")
    predict_lr_conf<-as.data.frame(predict_lr_conf)
    predict_lr_conf$predicted<-ifelse(predict_lr_conf$predict_lr_conf>0.4826,1,0)
    predict_lr_conf_final<-c(predict_lr_conf_final,predict_lr_conf)
    auc_lr_conf[i]<-auc(dataset$OUTPUT,predicted = predict_lr_conf$predicted)
    acc_lr_conf[i]<-mean(dataset$OUTPUT==predict_lr_conf$predicted)
    conf_lr_conf[i]<-mean(predict_lr_conf$predict_lr_conf)
    print("lr")
  
    #NN --------------
    #auc+acc
    # max_it<-4000
     scl <- function(x){ (x - min(x))/(max(x) - min(x)) }
     train_input<-train[,-16]
     train_input<- data.frame(lapply(train_input, scl))
     out_train<-class.ind(as.factor(train$OUTPUT))
     train_cv<-dataset[,-16]
     train_cv<- data.frame(lapply(train_cv, scl))
    # repeat
    # {
    #   nn_acc_auc <- nnet(x=train_input,y = out_train,data=train_cv,size= c(51), softmax=TRUE,maxit = max_it,MaxNWts = 1e100)
    #   if(nn_acc_auc$convergence==0 || max_it>10000)
    #   {
    #     break()
    #   }
    #   else
    #   {
    #     max_it<-max_it+3000
    #     print(nn_acc_auc$convergence)
    #   }
    # }
    # predict_nn_auc_acc<-predict(nn_acc_auc,dataset)
    # if(dim(predict_nn_auc_acc_final)[1]==0)
    # {
    #   predict_nn_auc_acc_final<-cbind(predict_nn_auc_acc)
    # }
    # else
    # {
    #   predict_nn_auc_acc_final<-cbind(predict_nn_auc_acc,predict_nn_auc_acc_final)
    # }
    # predict_nn_auc_acc_netresult<-as.data.frame(max.col(predict_nn_auc_acc)-1)
    # acc_nn_a[i]<-mean(predict_nn_auc_acc_netresult$`max.col(predict_nn_auc_acc) - 1`==dataset$OUTPUT)
    # auc_nn_a[i]<-auc(dataset$OUTPUT,predict_nn_auc_acc_netresult$`max.col(predict_nn_auc_acc) - 1`)
    # cnf<-ifelse(predict_nn_auc_acc[,1]>predict_nn_auc_acc[,2],predict_nn_auc_acc[,2],predict_nn_auc_acc[,1])
    # conf_nn_a[i]<-mean(cnf)
    # # confidence
    # max_it<-4000
    # repeat
    # {
    #   nn_conf <- nnet(x=train_input,y = out_train,data=train_cv,size= c(46), softmax=TRUE,maxit = max_it,MaxNWts = 1000)
    #   if(nn_conf$convergence==0 || max_it>10000)
    #   {
    #     break()
    #   }
    #   else
    #   {
    #     max_it<-max_it+3000
    #   }
    # }
    # predict_nn_conf<-predict(nn_conf,dataset)
    # if(dim(predict_nn_conf_final)[1]==0)
    # {
    #   predict_nn_conf_final<-cbind(predict_nn_conf)
    # }
    # else
    # {
    #   predict_nn_conf_final<-cbind(predict_nn_conf_final,predict_nn_conf)
    # }
    # predict_nn_conf_netresult<-as.data.frame(max.col(predict_nn_conf)-1)
    # acc_nn_conf[i]<-mean(predict_nn_conf_netresult$`max.col(predict_nn_conf) - 1`==dataset$OUTPUT)
    # auc_nn_conf[i]<-auc(dataset$OUTPUT,predict_nn_conf_netresult$`max.col(predict_nn_conf) - 1`)
    # cnf<-ifelse(predict_nn_conf[,1]>predict_nn_conf[,2],predict_nn_conf[,2],predict_nn_conf[,1])
    # conf_nn_conf[i]<-mean(cnf)
    # 
    # print("nn")
    # 
    # #two class SVM------------------  
    # model_TCSVM <- svm(OUTPUT~.,data = train,probability=TRUE) #train an two-classification model
    # pred_svm<- predict(model_TCSVM, dataset, probability=TRUE)
    # if(dim(pred_svm_final_class)[1]==0)
    # {
    #   pred_svm_final_class<-cbind(pred_svm)
    #   pred_svm_final_prob<-cbind(attr(pred_svm,"prob"))
    # }
    # else
    # {
    #   pred_svm_final_class<-cbind(pred_svm_final_class,pred_svm)
    #   pred_svm_final_prob<-cbind(pred_svm_final_prob,attr(pred_svm,"prob"))
    # }
    # acc_SVM[i]<-mean(pred_svm==dataset$OUTPUT)
    # auc_SVM[i]<-auc(dataset$OUTPUT,pred_svm)
    # cnf<-(ifelse(pred_svm==0,attr(pred_svm,"prob")[,2],attr(pred_svm,"prob")[,1]))
    # conf_svm[i]<-mean(cnf)
    # 
    # print("TCSVM")
    # 
    #adaboost----------
    model_ada<-ada(OUTPUT~.,train,loss="logistic",iter=50,nu=1, max.iter= 200)#best once
    pred_ada<-as.data.frame(predict(model_ada,dataset,type="both"))
    pred_ada_final<-c(pred_ada_final,pred_ada)
    acc_ada[i]<-mean(pred_ada$class==dataset$OUTPUT)
    auc_ada[i]<-auc(dataset$OUTPUT,pred_ada$class)
    cnf<-(ifelse(pred_ada$class==0,pred_ada$probs.2,pred_ada$probs.1))
    conf_ada[i]<-mean(cnf)
    print("ada")
    
    #C5.0 and CART --------------
    trainX<-train_input
    trainY<-as.factor(train$OUTPUT)
    model_c50<-C5.0(trainX,trainY,trials = 28)
    model_c50_win<-C5.0(trainX,trainY,trials = 28, control = C5.0Control(winnow = TRUE))
    model_rpart<-rpart(OUTPUT~.,data = train,method = "class")
    pred_c50<-as.data.frame(predict(model_c50,dataset,type = "prob"))
    pred_c50$class<-ifelse(pred_c50$`0`<pred_c50$`1`,0,1)
    acc_c50[i]<-mean(pred_c50$class==dataset$OUTPUT)
    auc_c50[i]<-auc(dataset$OUTPUT,pred_c50$class)
    cnf<-(ifelse(pred_c50$class==0,pred_c50$`1`,pred_c50$`0`))
    conf_c50[i]<-mean(cnf)
    pred_c50_final<-c(pred_c50_final,pred_c50)
    
    pred_c50_win<-as.data.frame(predict(model_c50_win,dataset,type = "prob"))
    pred_c50_win$class<-ifelse(pred_c50_win$`0`<pred_c50_win$`1`,0,1)
    pred_c50_win_final<-c(pred_c50_win_final,pred_c50_win)
    
    acc_c50_win[i]<-mean(pred_c50_win$class==dataset$OUTPUT)
    auc_c50_win[i]<-auc(dataset$OUTPUT,pred_c50_win$class)
    cnf<-(ifelse(pred_c50_win$class==0,pred_c50_win$`1`,pred_c50_win$`0`))
    conf_c50_win[i]<-mean(cnf)
    
    pred_rpart<-as.data.frame(predict(model_rpart,dataset,type = "prob"))
    pred_rpart$class<-ifelse(pred_rpart$`0`>pred_rpart$`1`,0,1)
    pred_rpart_final<-c(pred_rpart_final,pred_rpart)
    acc_rpart[i]<-mean(pred_rpart$class==dataset$OUTPUT)
    auc_rpart[i]<-auc(dataset$OUTPUT,pred_rpart$class)
    cnf<-(ifelse(pred_rpart$class==0,pred_rpart$`1`,pred_rpart$`0`))
    conf_rpart[i]<-mean(cnf)
    print("c50 cart")
    
    #KNN------------
    dummy<-dataset
    dummy$OUTPUT<-NULL
    pred_knn_36<-knn(trainX,dummy,trainY,36,prob = TRUE)
    pred_knn_23<-knn(trainX,dummy,trainY,23,prob = TRUE)
    pred_knn_23_class<-as.data.frame(as.numeric(pred_knn_23)-1)
    pred_knn_36_class<-as.data.frame(as.numeric(pred_knn_36)-1)
    if(dim(pred_knn_36_final_class)[1]==0)
    {
      pred_knn_36_final_class<-cbind(pred_knn_36_class$`as.numeric(pred_knn_36) - 1`)
      pred_knn_36_final_prob<-cbind(attr(pred_knn_36,"prob"))
      pred_knn_23_final_class<-cbind(pred_knn_23_class$`as.numeric(pred_knn_23) - 1`)
      pred_knn_23_final_prob<-cbind(attr(pred_knn_23,"prob"))
      
    }
    else
    {
      pred_knn_36_final_class<-cbind(pred_knn_36_final_class,pred_knn_36_class$`as.numeric(pred_knn_36) - 1`)
      pred_knn_36_final_prob<-cbind(pred_knn_36_final_prob,attr(pred_knn_36,"prob"))
      pred_knn_23_final_class<-cbind(pred_knn_23_final_class,pred_knn_23_class$`as.numeric(pred_knn_23) - 1`)
      pred_knn_23_final_prob<-cbind(pred_knn_23_final_prob,attr(pred_knn_23,"prob"))
    }
    acc_knn36_final[i]<-mean(pred_knn_36_class$`as.numeric(pred_knn_36) - 1`==dataset$OUTPUT)
    auc_knn36_final[i]<-auc(dataset$OUTPUT,pred_knn_36_class$`as.numeric(pred_knn_36) - 1`)
    cnf<-1-attr(pred_knn_36,"prob")#(ifelse(pred_knn_36==0,attr(pred_knn_36,"prob")[,2],attr(pred_knn_36,"prob")[,1]))
    conf_knn36_final[i]<-mean(cnf)
  
    acc_knn23_final[i]<-mean(pred_knn_23_class$`as.numeric(pred_knn_23) - 1`==dataset$OUTPUT)
    auc_knn23_final[i]<-auc(dataset$OUTPUT,pred_knn_23_class$`as.numeric(pred_knn_23) - 1`)
    cnf<-1-attr(pred_knn_23,"prob")#(ifelse(pred_knn_23==0,attr(pred_knn_23,"prob")[,2],attr(pred_knn_23,"prob")[,1]))
    conf_knn23_final[i]<-mean(cnf)
    
    print("knn")
    print(i)
    
    #correlation--------------
    output_matrix<-cbind(predict_lr_acc$predict_lr_acc)
    output_matrix<-cbind(output_matrix,predict_lr_auc$predict_lr_auc)
    output_matrix<-cbind(output_matrix,predict_lr_conf$predict_lr_conf)
  #  output_matrix<-cbind(output_matrix,predict_nn_auc_acc[,1])
  #  output_matrix<-cbind(output_matrix,predict_nn_conf[,1])
    output_matrix<-cbind(output_matrix,pred_ada$probs.1)
  #  output_matrix<-cbind(output_matrix,pred_c50$`0`)
  #  output_matrix<-cbind(output_matrix,pred_c50_win$`0`)
    output_matrix<-cbind(output_matrix,pred_rpart$`0`)
    output_matrix<-cbind(output_matrix,pred_rf$`0`)
    output_matrix<-cbind(output_matrix,attr(pred_knn_23,"prob"))
    output_matrix<-cbind(output_matrix,attr(pred_knn_36,"prob"))
    
  #  output_matrix<-cbind(output_matrix,attr(pred_svm,"prob")[,1])
    names(output_matrix)<-c("lr_acc","lr_auc","lr_conf","ada","cart","rf","knn23","knn36")
    if(dim(total_corr)[1]==0)
    {
      total_corr<-cbind(as.vector(cor(output_matrix)))
    }
    else
    {
      total_corr<-cbind(total_corr,as.vector(cor(output_matrix)))
    }
    
    
    output_matrix_class<-cbind(predict_lr_acc$predicted)
    output_matrix_class<-cbind(output_matrix_class,predict_lr_auc$predicted)
    output_matrix_class<-cbind(output_matrix_class,predict_lr_conf$predicted)
  #  output_matrix_class<-cbind(output_matrix,predict_nn_auc_acc_netresult)
  #  output_matrix_class<-cbind(output_matrix,predict_nn_conf_netresult)
    output_matrix_class<-cbind(output_matrix_class,pred_ada$class)
  #  output_matrix_class<-cbind(output_matrix_class,pred_c50$class)
  #  output_matrix_class<-cbind(output_matrix_class,pred_c50_win$class)
    output_matrix_class<-cbind(output_matrix_class,pred_rpart$class)
    output_matrix_class<-cbind(output_matrix_class,pred_rf$pred_rf)
    output_matrix_class<-cbind(output_matrix_class,as.numeric(pred_knn_23)-1)
    output_matrix_class<-cbind(output_matrix_class,as.numeric(pred_knn_36)-1)
    
  #  output_matrix_class<-cbind(output_matrix,pred_svm)
    names(output_matrix_class)<-c("lr_acc","lr_auc","lr_conf","ada","cart","rf","knn23","knn36")
    if(dim(total_corr_class)[1]==0)
    {
      total_corr_class<-cbind(as.vector(cor(output_matrix_class)))
    }
    else
    {
      total_corr_class<-cbind(total_corr_class,as.vector(cor(output_matrix_class)))
    }
  }
  #save.image("/home/swarnendu/Documents/final_ensemble_final_submission.Rdata")
  Stop<-Sys.time()
