This is a web app which predicts profit.
Its devloped using the dataset of 50_startup.
The notebook is avaiable is notebook folder

Technology Used : 
Docker
Flask
Machine Learning


How to Run it :
cd webapp 

#build docker 
#docker build  -t image_name .

docker build -t profitweb .

#run docker
docker run -p 5000:5000 profitweb

