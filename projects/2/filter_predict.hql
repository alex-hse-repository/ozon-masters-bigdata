ADD FILE predict.py;
INSERT INTO TABLE hw2_pred 
select TRANSFORM(*) USING 'python3 predict.py' from hw2_test
where 20 < f1 and f1 < 40;