****Travel Recommendation App****

A web-based application that suggests travel destinations based on user preferences for state, type of travel, best time to visit, and popularity rating.


**Features**

    Dynamic Destination Suggestions: Based on user input (state, type, best time, and popularity).
    Interactive Form: Users submit their preferences through a web form.
    Responsive Design: The layout adjusts for various screen sizes.

**Technologies Used**

    Frontend: HTML, CSS, JavaScript
    Backend: Flask (Python), Scikit-Learn
    Machine Learning: Decision Tree Classifier to suggest destinations

**Setup and Installation**

Clone the Repository:

        git clone `https://github.com/srilalita-05/Travel-recommendation-.git
        cd travel-recommendation`

*Install Dependencies*: Ensure Python 3 is installed, then install required packages:

      pip install flask pandas scikit-learn

*Add Dataset*: Place your dataset file (Expanded_Destinations.csv) in the project root directory.

*Run the Application:*

    python app.py

    Access the app in your browser at http://127.0.0.1:5000.

*Project Structure*

    app.py: Backend logic for handling user input, preprocessing, and recommending destinations based on a Decision Tree model.
    static/: Contains CSS (style.css) and JavaScript (script.js) files.
    templates/: HTML files, including index.html for the main page.
    Expanded_Destinations.csv: CSV dataset used for training the recommendation model.

*Usage*

    Select Preferences: Use the dropdowns to select a state, travel type, best time, and popularity rating.
    Get Recommendation: Click "Get Recommendation" to receive a destination suggestion tailored to your input.

**Example**

A user selects the following options:

    State: Goa
    Type of Travel: Beach
    Best Time to Visit: Nov-Feb
    Popularity: 8.5

The app will then return a recommended destination based on these inputs.


**License**

This project is licensed under the MIT License.


**Author**

    D Sri Lalita Bhuvaneswari 
