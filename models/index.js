const {Sequelize} = require('sequelize');

 const db= {};
 console.log("Here in index, connecting to the db");

 db.sequelize = new Sequelize("postgres://"+process.env.DB_USERNAME+":"+process.env.DB_PASS+"@ruverse.cvnpqsvbcz4r.ap-northeast-2.rds.amazonaws.com:5432/postgres",{
    logging:false,
    maxConcurrentQueries:800,
    dialect:"postgres",
    protocol:"postgres",
    dialectOptions:{
        ssl:{
            rejectUnauthorized:false
        },
        connectTimeout:7000
    }
  });
  
 db.Models=require("./model")(db.sequelize, Sequelize);
 module.exports={db};