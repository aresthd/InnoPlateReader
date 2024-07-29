# InnoPlateReader

InnoPlateReader is a Web-based system designed to detect and read vehicle license plates using images. InnoPlateReader implements the YOLOv8 (You Only Look Once) algorithm and Optical Character Recognition (OCR) to detect and read license plates. InnoPlateReader uses TailwindCSS technology as a front-end framework and Flask as a back-end framework. InnoPlateReader has also used Docker technology to deploy applications.


## Main Features
- **Home Page:** The start page of the InnoPlateReader system.
- **About Page:** Contains detailed information about InnoPlateReader.
- **Read Plate Page:** The page to perform the license plate detection process by uploading an image. The result of the detection will be displayed and the user can save the result.
- **Upgrade Model Page:** A page to upgrade the model by uploading a dataset containing images and labels (“.txt”), and specifying the epoch and confidence values. The system will display the training results such as confusion matrix and the user can save the new model results.


## Technologies Used
- **YOLOv8:** An object detection algorithm used to detect license plates.
- **OCR:** Technology for reading characters on license plates.
- **TailwindCSS:** Front-end framework used for website design and layout.
- **Flask:** A back-end framework used to manage server workflows and APIs.
- **Docker:** Platform for packaging, distributing, and running application in isolated container.


## Implementation Stages
The implementation stage of this project is as shown below.

![Implementation stages](/presentation/stages.jpg)


## Installation
How to install InnoPlateReader:
1. Clone this repository
   ```
   git clone https://github.com/AcilRestu12/InnoPlateReader.git
   cd InnoPlateReader
   ```
2. Setting path in /static/config.yaml
    ```
    path: <REPLACE WITH FULL PATH OF PROJECT>/static   # datset root dir
    train: images/train     # train images (relatives to 'path')
    val: images/train     # test images (relatives to 'path')
    
    # Classes
    names: 
        0: plat-nomor
    ```
3. Run your Docker
4. Build and run the Docker container
    ```
    docker-compose up
    ```
5. Access the app
    Open a browser and browse to http://localhost:5000.


Or you can get the application image by pull from Docker Hub:
1. Run your Docker
2. Pull the image
    ```
    docker pull acilrestu12/innoplatereader
    ```
3. Build and run the Docker container
    ```
    docker-compose up
    ```
4. Access the app:
    Open a browser and browse to http://localhost:5000.


## Guide
### Read Plate Page
1. Open the InnoPlateReader app, then click the Start Now button.
2. Open the “Read Plate” page on the website.
2. Upload an image of the vehicle with the license plate.
3. Click the “Process” button.
4. License plate detection results will be displayed.
5. You can save the detection result.

### Upgrade Model Page
1. Open the InnoPlateReader app, then click the Start Now button.
2. Open the “Upgrade Model” page on the website.
3. Upload the dataset of images and labels in .txt format.
4. Specify the preferred epoch and confidence values.
5. Click the “Train” button. Then wait for a while.
6. When finished, the system will display the confusion matrix of the model training results.
7. You can save the new model and use your model immediately.

## Structure of the Project
- **/static:** Asset files, images, css, and js.
- **/templates:** HTML template files.
- **/model:** The YOLOv8 model file used.
- **app.py:** The route file to set up the page logic.
- **static/config.yaml:** Config file path of the static folder
- **docker-compose.yml:** Docker Compose configuration file.
- **Dockerfile:** Docker configuration file for the application.
- **requirements.txt:** List of Python dependencies.


## Contribution Guidelines
1. Fork this repository.
2. Create a feature branch.
    ```
    git checkout -b new-feature
    ```
3. Commit your changes.
    ```
    git commit -am 'Add feature'
    ```
4. Push to the branch.
    ```
    git push origin new-feature
    ```
5. Make a Pull Request.


## License
The project is licensed under the MIT license - see the [LICENSE](LICENSE) file for more details.

## Note
This documentation includes a general description of the project, features provided, technologies used, installation instructions, how to use, project structure, contribution guidelines, and project license. You can further customize it according to the specific needs of the InnoPlateReader project.

