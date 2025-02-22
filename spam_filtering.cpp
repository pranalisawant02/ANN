import numpy as np
email ="you are selected for a job"
count=0
spam_words=0
#CHECK COUNT OF CAPITAL LETTERS
for i in email:
  if i.isupper():
    count+=1
capital_letter_count=count
#CHECK PRESENCE OF CERTAIN WORDS
certain_words =["buy", "cheap", "discount", "offer"]
for word in certain_words:
      if word in certain_words:
        if word in email.lower():
          spam_words+= 1;
          break
#check email length
if(len(email)<50):
  l=len(email)
else:
  l=0

x=np.array([capital_letter_count,l,spam_words])
y = np.array([1 if spam_words>0 or capital_letter_count>0 or l==0 else 0 ])

w=np.random.rand(3)
b=np.random.rand(1)

learning_rate=0.0001
iteration=1000

for i in range(iteration):

    y_predicted = w* x + b
    error=y_predicted - y

    dw=(2/len(x) * np.sum(error * x))
    db=(2/len(x) * np.sum(error))

    w = w - learning_rate * dw
    b = b - learning_rate * db

print(f'final weight:{w}')
print(f'final bias:{b}')
print(f'final error:{error}')
print(f'final predicted value:{y_predicted}')
print(f'final output:{y}')
