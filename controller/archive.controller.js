const dbContext=require("../models");
const fromServer = require("./../server");
const multer=require('multer')
const crypto=require("crypto");
var fs = require("fs");
var ffmpeg=require('fluent-ffmpeg');

var Archive=dbContext.db.Models[0];

 exports.create= (req,res) => 
 {
    dbContext.db.sequelize
    .authenticate()
    .then(()=>{
        console.log("Authenticated");
        dbContext.db.sequelize.sync();
    })
    .catch((err)=> {
        fromServer.aws=function(){
            return true;
        }
    })

    Archive=dbContext.db.Models[0]; 

    // var AWS=require("aws-sdk");
    // //UPLOADING FILE TO AWS S3
    // AWS.config.update({region: 'ap-northeast-2'});
    // s3 = new AWS.S3({apiVersion:"latest", accessKeyId:process.env.AWS_ACC_KEY, accessSecretKey:process.env.AWS_SECRET_KEY});


//UPLOAD

// let genId = '';
// const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
// const charactersLength = characters.length;
// let counter = 0;
// while (counter < 20) {
//   genId += characters.charAt(Math.floor(Math.random() * charactersLength));
//   counter += 1;
// }

//  var file = "test"+genId+".txt";

//  const uploadFile=(file,bucketName) => {
//     const fileContent = req.body.reduxData;
//     const params = {
//         Bucket: bucketName,
//         Key: file,
//         Body: fileContent,
//     }


// Configure the file stream and obtain the upload parameters LATER
// var fileStream = fs.createReadStream(file);
// fileStream.on("error", function (err) {
//   console.log("File Error", err);
// });
// uploadParams.Body = fileStream;

// var path = require("path");
// uploadParams.Key = path.basename(file);

// call S3 to retrieve upload file to specified bucket
// s3.upload(params, function (err, data) {
//   if (err) {
//     console.log("Error", err);
//   }
//   if (data) {
//     console.log("Upload Success", data.Location);
//   }
// });
//  }

//  uploadFile("test"+genId+".txt",'ruverse')

//  const uploadFile2=(file,bucketName) => {
//     console.log(req.files)
//     // const fileContent = fs.createReadStream(JSON.stringify(req.files[0].path).replace('"','').replace('"',''),'utf8');
//     // const fileContent=ffmpeg.ffprobe(JSON.stringify(req.files[0].path).replace('"','').replace('"',''))
//     const fileContent=fs.readFileSync(JSON.stringify(req.files[0].path).replace('"','').replace('"',''))
//     const params = {
//         Bucket: bucketName,
//         Key: file,
//         Body: fileContent,
//     }

// call S3 to retrieve upload file to specified bucket
// s3.upload(params, function (err, data) {
//   if (err) {
//     console.log("Error", err);
//   }
//   if (data) {
//     console.log("Upload Success", data.Location);
//   }
// });

//  uploadFile2("test"+genId+".mp4",'ruverse')


//     //////////////////////////////////////////////////////////////////////////
    console.log(req.body)
    constDataJSON = req.body
    Archive.create({uid:constDataJSON.uid, channelName:constDataJSON.channelName, loggerDataKey:constDataJSON.loggerDataKey, videoKey: constDataJSON.videoKey, screenKey: constDataJSON.screenKey})
    .then((data) =>{
        res.status(200).send(data)})
        .catch((err)=>res.status(400).send(err));
    }
    
    exports.findAll= (req,res) => 
    {
       dbContext.db.sequelize
       .authenticate()
       .then(()=>{
           console.log("Authenticated");
           dbContext.db.sequelize.sync();
       })
       .catch((err)=> {
           fromServer.aws=function(){
               return true;
           }
       })
   
       Archive=dbContext.db.Models[0]; 
       Archive.findAll().then((data)=>{
        res.send(data);
       }).catch((err)=>{
        res.status(500).send(err)
       })

       Archive=dbContext.db.Models[0]; 
       var AWS=require("aws-sdk");
//S3
       AWS.config.update({region: 'ap-northeast-2'});
       s3 = new AWS.S3({apiVersion:"latest", accessKeyId:process.env.AWS_ACC_KEY, accessSecretKey:process.env.AWS_SECRET_KEY});
   
       s3.listObjects({Bucket:'ruverse'}, function (err, data) {
           if (err) {
             console.log("Error", err);
           } else {
             console.log("Success", data);
           }
         });
       }


       exports.delete=(req,res) =>{
        dbContext.db.sequelize
        .authenticate()
        .then(()=>{
            console.log("Authenticated");
            dbContext.db.sequelize.sync();
        })
        .catch((err)=>{
            fromServer.aws=function(){
                return true
            }
        })
        Archive=dbContext.db.Models[0];
        Archive.destroy({
            where: {id:req.params.id}
        })
        res.status(204).json({
            msg:"Archive deleted"
        })
        .catch((err)=>{
            res.status(400).send({
                error:"Error while deleting"
            })
        })
       }

       exports.update=(req,res) =>{
        dbContext.db.sequelize
        .authenticate()
        .then(()=>{
            console.log("Authenticated");
            dbContext.db.sequelize.sync();
        })
        .catch((err)=>{
            fromServer.aws=function(){
                return true
            }
        })
        Archive=dbContext.db.Models[0];
        Archive.update(req.body,{
            where:{id:req.body.id}
        })
        .then((num)=>{
            if(num==1){
                res.status(200).send({
                    message:"Updated"
                })
            }
            else{
                res.status(400).send({
                    message:"Cannot update"
                })
            }
        })
       }

exports.charAnim=(req,res) => {
    var AWS=require("aws-sdk");
    AWS.config.update({
    accessKeyId: process.env.AWS_ACC_KEY,
    secretAccessKey: process.env.AWS_SECRET_KEY,
    region: 'ap-northeast-2'
  });
  
  // Create a new Polly object
  const polly = new AWS.Polly();
  
  // Function to generate speech audio using Amazon Polly
  async function generateSpeech(text, outputFilePath) {
    try {
      const params = {
        Text: text,
        OutputFormat: 'mp3',
        VoiceId: 'Joanna' // Specify the voice you want to use, e.g., Joanna, Matthew, etc.
      };
  
      // Request speech synthesis
      const data = await polly.synthesizeSpeech(params).promise();
  
      // Save audio data to file
      fs.writeFileSync(outputFilePath, data.AudioStream);
  
      console.log(`Speech generated successfully and saved to: ${outputFilePath}`);
    } catch (err) {
      console.error('Error generating speech:', err);
    }
  }
  
  // Example usage
  const text = 'Hello, this is a test.';
  
  const outputFilePath = 'output.mp3';
  
  generateSpeech(text, outputFilePath);

  const uploadFile=(file,bucketName) => {
    const fileContent=fs.readFileSync("./output.mp3")
    const params = {
        Bucket: bucketName,
        Key: file,
        Body: fileContent,
    }

AWS.config.update({region: 'ap-northeast-2'});
s3 = new AWS.S3({apiVersion:"latest", accessKeyId:process.env.AWS_ACC_KEY, accessSecretKey:process.env.AWS_SECRET_KEY});

// call S3 to retrieve upload file to specified bucket
s3.upload(params, function (err, data) {
  if (err) {
    console.log("Error", err);
    res.status(400).send({
      message:"Cannot upload to S3"
  })
  }
  if (data) {
    console.log("Upload Success", data.Location);
    res.status(200).send({
      message:"Uploaded successfully!"
  })
  }
});
 }

 uploadFile("output.mp3",'ruverse')
}
