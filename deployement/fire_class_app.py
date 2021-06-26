import streamlit as st 
# from PIL import Image
# import torch
# fig = plt.figure()

# with open("custom.css") as f:
# 	st.markdown(f"<style>{f.read()}<style>, unsafe_allow_html = True")

def main():
	''' A simple image classification program '''


	st.title('This is a Fire Classification Model')
	st.subheader('It will detect if the image has file')
	st.subheader('Please import the picture you want to classify')

	file_uploaded = st.file_uploader("Please choose a file :", type= ["png", "jpg", "jpeg"])
	if file_uploaded is not None:
		image = Image.open(file_uploaded)
		st.image(image, caption = 'Uploaded image', use_column_width=True)

		prediction = predict(image)
		print(f"Predicted value is {prediction.item()}")
		st.write(prediction)
		st.pyplot(fig)


def predict(image):
	google_dr = '/content/drive/MyDrive/Data_Science/projects/Fire_Detection_System/fire_classification/'
	saved_model = torch.load(google_dr +  'resnet50_model_6_23_full')

	# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
	inv_normalize = transforms.Normalize(
                    mean = [ -0.485/0.229, -0.456/0.224, -0.406/0.225], # -[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]
                    std  = [ 1/0.229, 1/0.224, 1/0.225])

	im = inv_normalize(image)
	is_fire = {1:'Fire', 0:'No_Fire'}


	#Evaluate the pic using mode
	saved_model.eval()
	im = im.to(devie) # not sure
	with torch.no_grad():
		prob = saved_model(im.view(1, 3, 224, 224))
		new_pred = saved_model(im.view(1, 3, 224, 224)).argmax().to(device)

	return new_pred


if __name__ == '__main__':
	main()

# https://www.youtube.com/watch?v=skpiLtEN3yk&ab_channel=JCharisTech
# https://blog.jcharistech.com/2019/10/24/how-to-deploy-your-streamlit-apps-to-heroku/

# to run on the local machine
# pipenv run streamlit run fire_class_app.py