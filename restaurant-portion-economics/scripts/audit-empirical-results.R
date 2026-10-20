# Audit existing Finnish outputs and original hotel/restaurant data.
# Run after scripts/download-data.R and scripts/analyse-finland.R.
source("R/finland.R")
source("R/empirical_audit.R")
dir.create("outputs",showWarnings=FALSE,recursive=TRUE)
x<-utils::read.csv("outputs/finland_daily.csv",stringsAsFactors=FALSE)
flags<-find_finland_discrepancies(x)
sensitivity<-finland_quality_sensitivity(x,flags)
utils::write.csv(flags,"outputs/finland_discrepancies.csv",row.names=FALSE)
utils::write.csv(sensitivity,"outputs/finland_sensitivity.csv",row.names=FALSE)
path<-"data/raw/urbanwaste/Food Tracking Device dataset.csv"
audit<-inspect_urbanwaste_csv(path)
utils::write.csv(audit$schema,"outputs/urbanwaste_schema.csv",row.names=FALSE)
cat("Flagged observations:",nrow(flags),"\n")
print(flags)
cat("Menu-matched sensitivity: descriptive only\n")
print(sensitivity)
cat("URBAN-WASTE rows:",nrow(audit$data),"fields:",ncol(audit$data),"\n")
print(audit$schema)
