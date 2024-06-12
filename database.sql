DROP DATABASE IF EXISTS sales_db;

CREATE DATABASE sales_db;
USE sales_db;

CREATE TABLE sales (
	id INT AUTO_INCREMENT PRIMARY KEY,
	sale_date DATE,
    product_id INT ,
    quantity INT,
    total_amount DECIMAL(10,2)
);

LOAD DATA 
    INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dataset.csv'
    INTO TABLE sales
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n'
    IGNORE 1 LINES
    (sale_date, product_id, quantity, total_amount);

SELECT * FROM sales;

