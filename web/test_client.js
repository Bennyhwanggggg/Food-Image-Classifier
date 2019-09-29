const fs = require('fs');
const request = require('request');

var formData = {
    file: fs.createReadStream('testimg.jpeg'),
};

request.post({url:'http://0.0.0.0:5000/predict', formData: formData}, function optionalCallback(err, httpResponse, body) {
    if (err) {
      return console.error('upload failed:', err);
    }
    console.log('Upload successful!  Server responded with:', body);
});