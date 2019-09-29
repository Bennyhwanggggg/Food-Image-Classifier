const fs = require('fs');
const request = require('request');

var formData = {
    file: fs.createReadStream('test3.jpeg'),
};

herokuurl = "https://food-predictor-ai.herokuapp.com/predict";
localurl = 'http://0.0.0.0:5000/predict';

request.post({url:herokuurl, formData: formData}, function optionalCallback(err, httpResponse, body) {
    if (err) {
      return console.error('upload failed:', err);
    }
    console.log('Upload successful!  Server responded with:', body);
});