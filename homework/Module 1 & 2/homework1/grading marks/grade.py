class grading:
    def __init__ (self, name, math_marks, sci_marks, eng_marks):
        self.name = name
        self.marks_data = {
            'Math' : math_marks,
            'Science' : sci_marks,
            'English' : eng_marks
        }
    def total_marks(self):
        return sum(self.marks_data.values())
    def __str__(self):
        return f"Student: {self.name}, Marks: {self.marks_data}, Total: {self.total_marks()}"
    
student_names = []
for i in range(3):
    student_names.append(input(f"Enter the name of the student {i+1}: "))

all_students = []

for name in student_names:
    print(f"Entering marks for {name}")

    while True:
        try:
            math_marks = int(input(f"Enter marks for {name} in Math: "))
            if 0<= math_marks <=100:
                break
            else: 
                print("ERROR: Enter marks between 0-100.")
        except ValueError:
            print("Invalid input, please enter a valid number.")

    while True:
        try:
            sci_marks = int(input(f"Enter marks for {name} in Science: "))
            if 0<=sci_marks<=100:
                break
            else:
                print("ERROR: Enter marks between 0-100.")
        except ValueError:
            print("Invalid input, please enter a valid number.")
    while True:
        try:
            eng_marks = int(input(f"Enter marks for {name} in English: "))
            if 0<=eng_marks<=100:
                break
            else:
                print("ERROR: Enter marks between 0-100.")
        except ValueError:
            print("Invalid input, please enter a valid number.")
    new_student = grading(name,math_marks,sci_marks,eng_marks)
    all_students.append(new_student)


for student in all_students:
    print(student)

best_student = max(all_students, key= lambda s: s.total_marks())
print (f"The student with the highest marks is {best_student.name} with a total marks of {best_student.total_marks()}")