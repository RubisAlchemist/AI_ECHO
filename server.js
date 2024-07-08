const http=require('http');
const https=require('https');

aws=function(){
    return false;
}
exports.aws=aws;

const express=require('express');
const cors=require('cors');
const session=require('express-session');
require("dotenv").config();
const db=require('./models/index');
const archiveRouter = require("./routes/archive.routes");
const archiveController=require("./controller/archive.controller");

const fs=require('fs');
const path=require('path');
const bodyParser = require('body-parser');

var AWS = require('aws-sdk');

const app=express();
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended:false}))

var corsOpt={
    origin:true,
    methods:['GET','PUT','POST','DELETE','FETCH','OPTIONS'],
    preflightContinue:true,
    allowedHeaders:['Content-Type','Authorization','Content-length','X-Requested-With','Accept'],
    credentials:true
}
app.options('*',cors(corsOpt));
app.use(cors(corsOpt));


db.db.sequelize
.authenticate()
.then(()=>{
    console.log("Authenticated");
    db.db.sequelize.sync().then(()=>{
        console.log("Success")});
})

app.use("/", archiveRouter);

const options={
    key:fs.readFileSync("./private.key"),
    cert:fs.readFileSync("./certificate.crt"),
    ca:fs.readFileSync("./ca_bundle.crt")
}

https.createServer(options,app).listen(443,function(req,res){
    console.log("Server started on port 443")
});

// http.createServer(app).listen(8080,function(req,res){
//     console.log("Server started on port 8080")
// })

// const options={
//     key:fs.readFileSync("./private/key"),
//     cert:fs.readFileSync("./certificate.crt"),
//     ca:fs.readFileSync("./ca_bundle.crt")
// }
// https.createServer(options,app).listen(443,function(req,res){
//     console.log("Server started on port 443")
// });

