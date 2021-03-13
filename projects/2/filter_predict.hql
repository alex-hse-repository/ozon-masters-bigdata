ADD FILE projects/2/predict.py;
INSERT INTO TABLE hw2_pred 
select TRANSFORM(*) USING 'python3 predict.py' from 
(select * from hw2_test
where (f1 is not NULL) and (20 < f1 and f1 < 40));