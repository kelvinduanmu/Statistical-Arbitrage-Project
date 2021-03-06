library(table)
library(xts)
dir="/Users/jenniferfu/Documents/MSCF/2017-18/Statistical Arbitrage/Project/Data"
setwd(dir)
MKT_CAP=read.csv("MKT_CAP.csv",header=TRUE,na.strings = c("NA","Inf"))
PX_LAST=read.csv("PX_LAST.csv",header=TRUE,na.strings = c("NA","Inf"))
PX_LAST=PX_LAST[,-which(colnames(PX_LAST)=="MBAY")]
MKT_CAP=MKT_CAP[,-which(colnames(MKT_CAP)=="MBAY")]
################################################
##Price
PX.rev=PX_LAST[seq(dim(PX_LAST)[1],1),-1]
row.names(PX.rev)=seq(1:nrow(PX_LAST))
index=apply(PX.rev,2,function(x) min(which(x!=x[1])))
index=index-2
for(i in 1:ncol(PX.rev)){
  PX.rev[1:(index[i]),i]=NA
}
PX.clean=PX.rev[seq(dim(PX.rev)[1],1),]
PX.clean=PX.clean[-nrow(PX.clean),]
Dates=as.Date(PX_LAST[,1])[-nrow(PX_LAST)]
PX.clean=as.xts(PX.clean,Dates)

###MarketCap
a=MKT_CAP[-nrow(MKT_CAP),-1]
naindex=!is.na(PX.clean)
naindex=replace(naindex,naindex==0,NA)
MCAP.Clean=a*naindex
MCAP.Clean=as.xts(MCAP.Clean,Dates)

###MarketShare
MKshare=t(apply(MCAP.Clean,1,function(x) x/sum(x,na.rm = T))) 
MKshare=as.xts(MKshare)

##Benchmark
stock.daily.ret=diff(PX.clean)/rbind(rep(1,ncol(PX.clean)),as.matrix(PX.clean[-nrow(PX.clean),]))
bm.ret=rowSums(stock.daily.ret*MKshare,na.rm = T)
bm.ret=as.xts(bm.ret,Dates)
bm.val=cumprod(bm.ret+1)
bm.val=as.xts(bm.val,Dates)
plot.ts(bm.val)

#weeklydata
WeeklyDates=Dates[weekdays(Dates)=="Friday"]
MKshare.weekly = MKshare[WeeklyDates,]
PX.weekly=PX.clean[WeeklyDates,]
stock.weekly.ret=diff(PX.weekly)/rbind(rep(1,ncol(PX.weekly)),as.matrix(PX.weekly[-nrow(PX.weekly),]))
bm.weekly.ret=rowSums(stock.weekly.ret*MKshare.weekly,na.rm = T)
bm.weekly.ret=as.xts(bm.weekly.ret,WeeklyDates)
colnames(bm.weekly.ret)="bm_weekly"
bm.weekly.val=cumprod(bm.weekly.ret+1)
plot.ts(bm.weekly.val)

#annualize 
stock.ret.annu=(stock.weekly.ret+1)^52-1
bm.ret.annu=(bm.weekly.ret+1)^52-1
##rf
rf.ts=read.csv("WGS10YR.csv",header=T)
rf.ts=as.xts(rf.ts[,2],as.Date(rf.ts$Dates,format = "%m/%d/%Y"))
colnames(rf.ts)="rf"
rf.ts=rf.ts[WeeklyDates]*.01
##excess ret
stock.excess.ret=apply(stock.ret.annu,2,function(x) x-as.numeric(rf.ts))
bm.excess.ret=bm.ret.annu-as.numeric(rf.ts)
stock.excess.ret=as.xts(stock.excess.ret,WeeklyDates)

#regression
beta=matrix(rep(NA,length(stock.excess.ret)),ncol=500)
for(i in (1:ncol(stock.excess.ret))){
  data=as.data.frame(cbind(stock.excess.ret[,i],bm.excess.ret))
  start=(which(!is.na(data[,1]))[1])
  end=(dim(data)[1]-which(!is.na(rev(data[,1])))[1]+2)
  data=data[start:(end-1),]
  colnames(data)=c("y","x")
  beta[(start+104-1):(end-1),i]=rollapply(data, width = 104,
            FUN = function(z) coef(lm(y ~ x, data = as.data.frame(z),na.action = na.exclude))[2],
            by.column=FALSE, align="right")
}
beta=as.xts(beta,WeeklyDates)
colnames(beta)=ticker
##Reform
# ticker=colnames(PX.weekly)
# Data=as.data.frame(PX.weekly)
# colnames(Data) <- NULL
# 
# Data=as.data.frame(rbind(c("variable",ticker),as.data.frame(Data)))
# Data[1,1]="Dates"
# Data=cbind(rep("PX.weekly",778),as.data.frame(PX.weekly))
# colnames(Data)[1]="Variables"
# D=cbind(rep("stock.ret",778),as.data.frame(stock.ret.annu))
# colnames(D)[1]="Variables"
# Data=rbind(as.data.frame(Data),as.matrix(D))
# Data=rbind(as.array(colnames(Data)),as.data.frame(Data))

library(ggplot2)
library(reshape2)
test=as.data.frame(cbind(as.Date.character(WeeklyDates),scale(beta[,1:50],center = T,scale = T)))
colnames(test)[1]="Date"
test.mt <- melt(test, id="Date") 
ggplot(data=test.mt,aes(x=Date, y=value, colour=variable))+geom_line(show.legend=F)


write.csv(PX.weekly,file="/Clean/PX.Weekly.csv",row.names = WeeklyDates)
write.csv(stock.ret.annu,file="/Clean/stock.ret.csv",row.names = WeeklyDates)
write.csv(stock.excess.ret,file="/Clean/excess.ret.csv",row.names = WeeklyDates)
write.csv(beta,file="/Clean/beta.csv",row.names = WeeklyDates)

BM=cbind(bm.val=bm.weekly.val,bm.ret=bm.ret.annu,rf=rf.ts,bm.excess.ret=bm.excess.ret)
colnames(BM)=c("bm.val","bm.ret","rf","bm.excess.ret")
write.csv(BM,file="Clean/Benchmark.csv",row.names = WeeklyDates)
