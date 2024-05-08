import streamlit as st
import os
from streamlit_option_menu import option_menu

# Function to load image or animation
def load_media(selection):
    media_path = os.path.join("images", f"{selection}.jpg")
    if not os.path.exists(media_path):
        media_path = os.path.join("images", f"{selection}.gif")
    return media_path

# Page functions
def page_home():
    #load assets
    video1 = open("Video_call.mp4", "rb") 

    st.write("##")
    st.write("##")

    #header
    with st.container():
        st.write("Step into the world of Sign Language, where every sign tells a story and every gesture bridges worlds.:wave:")
        st.title("SignSpeak: Where silent gestures find their powerful voice.")
        st.write("##")
        st.subheader("Explore our project to witness the transformative impact of bridging communication divides and fostering inclusivity.")
        st.write("##")
        st.write("Join us as we embark on this enlightening journey together! [learn more >](https://google.com)")


    st.write("##")
    st.write("##")
    st.write("##")
    #what I do//// maybe add a video
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("What we do")
            st.write("##")
            st.subheader(
        """
        Hello, I'm [Your Name]. Welcome to signSpeak!
        
        - With signSpeak, we've created a platform that understands your signs, translates them into written text, and even converts them into spoken words.
        - Record your signs using a camera or webcam, and our technology will detect and translate them into your preferred language.
        - You can also convert the translated text into spoken words, making it easier for everyone to understand.
        
        Thank you for joining us on this journey. Feel free to reach out if you have any questions or feedback.

        """
            )
            
        with right_column:
            st.video(video1) #displaying the video

        # Add your home page content here

def page_model():
    st.write("This is the Model Page!")
    # Add your model page content here

def page_dictionary():
    st.title("Dictionary Page")
    st.write("This is the dictionary page.")
    
    st.write("Here you can find signs or explanations for selected alphabets/numbers.")

    
    # Define terms and their corresponding images
    images = {
        'A': 'A',
        'B': 'B',
        'C': 'C',
        'D': 'D',
        'E': 'E',
        'F': 'F',
        'G': 'G',
        'H': 'H',
        'I': 'I',
        'J': 'J',
        'K': 'K',
        'L': 'L',
        'M': 'M',
        'N': 'N',
        'O': 'O',
        'P': 'P',
        'Q': 'Q',
        'R': 'R',
        'S': 'S',
        'T': 'T',
        'U': 'U',
        'V': 'V',
        'W': 'W',
        'X': 'X',
        'Y': 'Y',
        'Z': 'Z',
        # Add more letters as needed
    }

    
    # Create four columns for buttons
    cols = st.columns(7)

    # Variable to store the selected letter
    selected_letter = None

    # Iterate over each letter and add a button to the corresponding column
    for i, (letter, _) in enumerate(images.items()):
        if i < 26: # limit to first 26 letters
            button = cols[i % 7].button(letter)
            if button:
                # Store the selected letter
                selected_letter = letter

    # Display image for the selected letter
    if selected_letter:
        # Display image for the selected option
        image_path = load_media(images[selected_letter])
        st.image(image_path, use_column_width=True)


def page_about_us():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Inspiration about the project")
        st.write("##")
        st.write(
            """
            The Idea of our project was:
            - blablabla
            - blablabla
            - blablabla
            - blablabla

            If you got any quesiton, feel free to connect with us through email.
            """
        )

#navbar
selected = option_menu(None, ["Home", "Model", "Dictionary", 'About Us'], 
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

# Page dictionary
pages = {
    "Home": page_home,
    "Model": page_model,
    "Dictionary": page_dictionary,
    "About Us": page_about_us,
}

# Run selected page
pages[selected]()