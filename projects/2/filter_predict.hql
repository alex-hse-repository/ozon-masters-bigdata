ADD FILE projects/2/predict.py;
ADD FILE projects/2/2.joblib;
INSERT INTO TABLE hw2_pred
select TRANSFORM(*) USING '/opt/conda/envs/dsenv/bin/python predict.py' from hw2_test
where (if1 is not NULL) and (20 < if1 and if1 < 40);



