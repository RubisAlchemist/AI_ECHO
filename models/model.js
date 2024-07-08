module.exports=(sequelize,Sequelize) => {
    const Archive=sequelize.define(
        "Archive",
        {
            uid:{
                type:Sequelize.STRING,
                onDelete:'CASCADE',
            },
            channelName:{
                type:Sequelize.STRING,
                allowNull:false,
            },
            loggerDataKey:{
                type:Sequelize.STRING,
                allowNull:false,
            },
            videoKey:{
                type:Sequelize.STRING,
                allowNull:false,

            },
		screenKey:{
		type:Sequelize.STRING,
		allowNull:true,
	    }},
        {
            tableName:"Archive",
        }
    )

sequelize.sync({alter:true});
return [Archive];}
