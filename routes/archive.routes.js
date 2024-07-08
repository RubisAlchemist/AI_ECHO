const express=require("express");
const multer=require('multer')
var fs = require("fs");
const upload=multer({limits:{fieldSize:25*1024*1024},  storage:multer.diskStorage({
    destination:(req,file,cb)=>{
        fs.mkdir('./uploads', (err)=>
            cb(null, './uploads'))
        },
        filename:(req,file,cb)=>{
            cb(null, file.originalname)
        }}
        
        )
    }
)

// , function(req,res){
// if(!req.file){
//     console.log('no file')
//     res.send({success:false})
// }
// else{
//     console.log('file')
//     res.send({success:true})
// }
// };
const archiveController = require("./../controller/archive.controller");
const { fstat } = require("fs");
const { ConnectContactLens } = require("aws-sdk");
const router=express.Router();

router.get("/",archiveController.findAll);
router.post("/upload",upload.any(), archiveController.create);
router.delete("/:id", archiveController.delete);
router.put("/:obj", archiveController.update)
router.get("/charAnim",upload.any(),archiveController.charAnim);

module.exports=router;