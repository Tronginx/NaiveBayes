# CSE 3521 HW3 Programming 

Name: Tron Tian

Date: 03/30/2022

### 1.output from NaiveBayes.py

![NaiveBayes](C:\SP22\CSE3521\HW\3\NaiveBayes\NaiveBayes.PNG)

### 2.output from NaiveBayes_improved.py![improved](C:\SP22\CSE3521\HW\3\NaiveBayes\improved.PNG)

### 3. What I did in NaiveBayes_improved.py

Since the error states we shouldn't "divide by zero". Then we may want to focus on the fraction part of our code. I added 1s to both side of the fraction as the code snippets shown below to fix the error.

```python
#calculate probability           
for i in range(3):
     self.conditional[i] = (self.conditional[i]+1)/(self.prior[i]+1)
        
self.prior = (self.prior+1) / (sum(self.prior)+1)
```

