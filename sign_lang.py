import streamlit as st
#pyttsx3
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
import requests  # pip install requests
from streamlit_option_menu import option_menu  # pip install streamlit-option-menu
# from main import main
from ww import main
st.set_page_config(page_title="Sign Language Helper", page_icon=":raised_hand_with_fingers_splayed:", layout="wide")
# st.write("helooooo")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
# video1 = open("sign_video.mp4", "rb")

# Navbar
selected = option_menu(None, ["Home", "Model", "Dictionary", "About Us"],
                       icons=["house", "cloud-upload", "list-task", "gear"],
                       menu_icon="cast", default_index=0, orientation="horizontal")

# Display content based on selected option
if selected == "Home":
    with st.container():
        st.write(
            "Step into the world of Sign Language, where every sign tells a story and every gesture bridges worlds.:wave:")
        st.title(" HearingHands : Amplifying the Voices of the Deaf Community")
        st.subheader(
            "Explore our project to witness the transformative impact of bridging communication divides and fostering inclusivity.")

    # what I do//// maybe add a video
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("What we do")
            st.write("##")
            st.subheader(
                """
                Hello, Welcome to HearingHands!
        
                - With HearingHands, we've created a platform that understands your signs, translates them into written text, and even converts them into spoken words.
                - Record your signs using a camera or webcam, and our technology will detect and translate them into your preferred language.
                - You can also convert the translated text into spoken words, making it easier for everyone to understand.
        
                Thank you for joining us on this journey. Feel free to reach out if you have any questions or feedback.
        
                """
            )

        with right_column:
            width = 46
            #     width = st.sidebar.slider(
            #     label="Width", min_value=0, max_value=100, value=DEFAULT_WIDTH, format="%d%%"
            # )
            #     width = max(width, 0.01)
            side = max((100 - width) / 2, 0.01)

            _, container, _ = st.columns([side, width, side])
            container.image(image="logo.png",width=375)

elif selected == "Model":
    main()
        
    

elif selected == "Dictionary":

    text = """
    Indiansignlanguage.org offers a vast collection of Indian Sign Language (ISL) signs. Each sign is accompanied by an image, a running video, and threaded discussions. It is an ideal resource for learning or teaching Indian Sign Language. We are continuously adding more signs and developing new services to empower the Deaf community.
    Please share your ideas and comments to help us improve this service."""


    # YouTube video URL
    video_url = "https://youtu.be/xwGHOUcnnHU?list=TLGGWlgSddFPLisyODAzMjAyNA"

    # Add some padding
    st.write("")
    st.write("")
    

    # Create columns for the layout
    col1, col2 = st.columns([4, 3])

    # Display text in the left column
    with col1:
        st.write("##")
        st.markdown(f'<div class="text-container">{text}', unsafe_allow_html=True)

    # Display YouTube video in the right column
    with col2:
        st.video(video_url)

    with st.sidebar:
        option = st.selectbox(
            "Which signs you would like to see?",
            ("Alphabets (A to Z)", "Numbers (0 to 9)"),
            index=0,
            # placeholder="Select one option"
        )

    if option == "Alphabets (A to Z)":
        num_columns = 5
        num_rows = 6  # 26 letters divided by 5 columns = 5.2 rows, rounded up to 6

        # Create a list to hold the content for each column
        columns = st.columns(num_columns)

        # Define the maximum number of images
        max_images = min(26, num_rows * num_columns)

        # Loop through each image
        for img_number in range(1, max_images + 1):
            # Calculate the row and column for the current image number
            row = (img_number - 1) // num_columns
            col_index = (img_number - 1) % num_columns

            # Get the column corresponding to the image's column index
            col = columns[col_index]

            # Display the image inside an expander
            with col.expander(f"Image {img_number}"):
                col.image(f"images/{img_number}.jpg", width=150)

        # Apply responsive CSS styling
        col_width = 100 / num_columns
        st.write(
            f"""
            <style>
            .reportview-container .main .block-container {{
                max-width: 100%;
            }}
            .reportview-container .main {{
                padding-left: 0;
                padding-right: 0;
            }}
            .reportview-container .main .block-container .block-element {{
                width: calc({col_width}% - 20px);
                display: inline-block;
                margin-right: 20px;
                margin-bottom: 20px;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )


    elif option == "Numbers (0 to 9)":
        num_columns = 5
        num_rows = 2

        # Create a list to hold the content for each column
        columns = st.columns(num_columns)

        # Loop through each row
        for row in range(num_rows):
            # Loop through each column
            for i, col in enumerate(columns):
                # Calculate the image number based on the row and column
                if row == 0:
                    img_number = row * num_columns + i + 30                 
                else:
                    img_number = (row - 1) * num_columns + i + 35
                # Display the image inside an expander
                with col.expander(f"Image {img_number}"):
                    # Check if the image number exists
                    if img_number <= 39:
                        col.image(f"images/{img_number}.jpg", width=200)

                        col.write(f"<div style='text-align: center; padding: 5px;'>{img_number - 30} </div>",
                                  unsafe_allow_html=True)

        # Apply responsive CSS styling
        col_width = 100 / num_columns
        st.write(
            f"""
            <style>
            .reportview-container .main .block-container {{
                max-width: 100%;
            }}
            .reportview-container .main {{
                padding-left: 0;
                padding-right: 0;
            }}
            .reportview-container .main .block-container .block-element {{
                width: calc({col_width}% - 20px);
                display: inline-block;
                margin-right: 20px;
                margin-bottom: 20px;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )





elif selected == "About Us":
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("Inspiration about the project")
            st.write("##")
            st.write(
                """
                The Idea of our project was:
                - To promote inclusive communication and support sign language users.
                - To break communication barriers between the deaf community.
                - Empower users to confidently express themselves and engage with others.

                If you got any quesiton, feel free to connect with us through email.
                """
            )
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")

    with st.container():
        st.write("---")
        st.header("Get In Touch With Us!")
        st.write("##")

        # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
        contact_form = """
        <form action="https://formsubmit.co/saeeshweta1831@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here" required></textarea>
            <button type="submit">Send</button>
        </form>
        """
        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown(contact_form, unsafe_allow_html=True)
        with right_column:
            st.empty()

