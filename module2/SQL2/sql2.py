# solutions.py
"""Volume 3: SQL 2.
R Scott Collings
Math 321 Sec 002
29 Oct 2017
"""

import sqlite3 as sql
import numpy as np

# Problem 1
def prob1(db_file="students.db"):
    """Query the database for the list of the names of students who have a
    'B' grade in any course. Return the list.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): a list of strings, each of which is a student name.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #find students who have any B only
            cur.execute("""select StudentName from StudentInfo as si 
                            inner join StudentGrades as sg on si.StudentID = sg.StudentID
                            where sg.Grade = 'B';""")
            matches =  cur.fetchall()
            return [match[0] for match in matches]
    finally:
        conn.close()



# Problem 2
def prob2(db_file="students.db"):
    """Query the database for all tuples of the form (Name, MajorName, Grade)
    where 'Name' is a student's name and 'Grade' is their grade in Calculus.
    Only include results for students that are actually taking Calculus, but
    be careful not to exclude students who haven't declared a major.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #innermost select gets info for all grades in calculus course
            #middle select gets student info for all people with grades in calc courses
            #final select returns majors of these students and their grades
            cur.execute("""select StudentName as Name, MajorName, Grade
                            from (
                                select * from StudentInfo as si 
                                inner join (
                                    select * from StudentGrades as sg 
                                    inner join CourseInfo as ci 
                                    on sg.CourseID = ci.CourseID 
                                    where ci.CourseName = 'Calculus'
                                ) as calc
                            on calc.StudentID = si.StudentID) as calcstuds
                            left join MajorInfo as mi 
                            on calcstuds.MajorID = mi.MajorID;
                        """)
            return cur.fetchall()
    finally:
        conn.close()


# Problem 3
def prob3(db_file="students.db"):
    """Query the database for the list of the names of courses that have at
    least 5 students enrolled in them.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        ((list): a list of strings, each of which is a course name.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #courses that have 5 or more students enrolled
            cur.execute("""
                        select count(StudentID), CourseName from
                        StudentGrades as sg inner join CourseInfo as ci
                        on sg.CourseID = ci.CourseID
                        group by CourseName
                        having count(*) >= 5;
                        """)
            matches = cur.fetchall()
            return [match[1] for match in matches]
    finally:
        conn.close()


# Problem 4
def prob4(db_file="students.db"):
    """Query the given database for tuples of the form (MajorName, N) where N
    is the number of students in the specified major. Sort the results in
    descending order by the counts N, then in alphabetic order by MajorName.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #Popularity of majors
            cur.execute("""
                        select MajorName, count(StudentID) as N
                        from StudentInfo as si left join MajorInfo as mi
                        on mi.MajorID = si.MajorID
                        group by MajorName
                        order by N asc;
                        """)
            return cur.fetchall()
    finally:
        conn.close()


# Problem 5
def prob5(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, MajorName) where
    the last name of the specified student begins with the letter C.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #left join so students with C last names that don't have major show up
            cur.execute("""
                        select StudentName, MajorName
                        from StudentInfo as si left join MajorInfo as mi
                        on mi.MajorID = si.MajorID
                        where StudentName like '% C%';
                        """)
            return cur.fetchall()
    finally:
        conn.close()


# Problem 6
def prob6(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, N, GPA) where N
    is the number of courses that the specified student is in and 'GPA' is the
    grade point average of the specified student according to the following
    point system.

        A+, A  = 4.0    B  = 3.0    C  = 2.0    D  = 1.0
            A- = 3.7    B- = 2.7    C- = 1.7    D- = 0.7
            B+ = 3.4    C+ = 2.4    D+ = 1.4

    Order the results from greatest GPA to least.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #inner select gives students and the gpa for each course
            #outer select counts courses and averages gpa points
            cur.execute("""
                        select StudentName, count(sc.points), avg(sc.points) as GPA
                        from StudentInfo as si inner join (
                            select StudentId, case Grade 
                            when 'A+' then 4.0
                            when 'A' then 4.0
                            when 'A-' then 3.7
                            when 'B+' then 3.4
                            when 'B' then 3.0
                            when 'B-' then 2.7
                            when 'C+' then 2.4
                            when 'C' then 2.0
                            when 'C-' then 1.7
                            when 'D+' then 1.4
                            when 'D' then 1.0
                            when 'D-' then 0.7 end as points
                            from StudentGrades as sg inner join CourseInfo as ci
                            on sg.CourseID = ci.CourseID
                        ) as sc
                        on si.StudentID = sc.StudentID
                        group by StudentName
                        order by GPA desc;
                        """)
            return cur.fetchall()
    finally:
        conn.close()
