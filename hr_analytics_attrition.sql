use hr_analytics_attrition;

-- check all of the data
SELECT * FROM attrition;

# 1
-- what is the relationship between attrition and job satisfaction
SELECT Attrition, JobSatisfaction, COUNT(*) AS Count
FROM attrition
GROUP BY Attrition, JobSatisfaction
ORDER BY Attrition, JobSatisfaction;
# we can see that the higher the job satisfaction, the less likely it is for someone to leave

# 2
-- what is the relationship between attrition and gender
SELECT 
	a.Attrition, 
	a.Gender, 
	COUNT(*) AS Count, 
	ROUND((COUNT(*) / t.total_count) * 100, 2) AS Percentage
FROM 
	attrition AS a
JOIN
	(
		SELECT
			Gender,
            COUNT(*) AS total_count
		FROM 
			attrition
		GROUP BY
			Gender
	) AS t
ON
	a.GENDER = t.Gender
GROUP BY a.Attrition, a.Gender
ORDER BY a.Attrition, a.Gender;
# We can see that the proportion of mena nd women that leave vs stay is very similar

# 3
-- calculate the employee attrition rate

-- first calculate the number of employees who left
SELECT COUNT(*) AS left_company
FROM attrition
WHERE Attrition = 'Yes';

-- then calculate the total nmber of employees
SELECT COUNT(*) AS total_employees
FROM attrition;

-- now we put it all together to calculate the attrition rate
SELECT ROUND((left_company/total_employees) * 100, 2) AS attrition_rate
FROM(
	SELECT COUNT(*) AS left_company
    FROM attrition
    WHERE Attrition = 'Yes'
) AS count_of_left,
(
	SELECT COUNT(*) AS total_employees
    FROM attrition
) AS total_count;
# The attrition rate is 16%