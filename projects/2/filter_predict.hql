ADD FILE projects/2/model.py;
ADD FILE projects/2/predict.py;
ADD FILE 2.joblib;
INSERT INTO TABLE hw2_pred
select * from (select TRANSFORM(*) USING 'predict.py' as (id,pred) from hw2_test
where (if1 > 20 and if1 < 40)) tmp;




