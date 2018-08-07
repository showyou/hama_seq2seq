#!/usr/bin/env python
# -*- coding: utf-8 -*-

# マルコフテーブル等定義
import sqlalchemy
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import Column, types
from sqlalchemy.ext.declarative import declarative_base
import datetime
Base = declarative_base()

class Tweet(Base):
    __tablename__ = 'tweet'

    id = Column(types.Integer, primary_key=True)
    user = Column(types.Unicode(32))
    text = Column(types.Unicode(140))
    datetime = Column(types.DateTime, default=datetime.datetime.now())
    replyID = Column(types.String(64), default=-1)
    isAnalyze = Column(types.SmallInteger, default=False)


class OhayouTime(Base):
    __tablename__ = 'ohayouTime'
    
    id = Column(types.Integer, primary_key=True)
    user = Column(types.Unicode(32))
    type = Column(types.Unicode(32))
    datetime = Column(types.DateTime, default=datetime.datetime.now())


class Reply(Base):
    __tablename__ = 'reply'

    id = Column(types.Integer, primary_key=True)
    tweet_id = Column(types.BigInteger())
    reply_text = Column(types.Text)
    src_id = Column(types.BigInteger())
    src_text = Column(types.Text)
    is_analyze = Column(types.SmallInteger, default=False) 

    def __repr__(self):
        return "<Reply(tweet_id='%d', reply_text='%s', src_id='%s', src_text='%s')>" % (
                self.tweet_id, self.reply_text, self.src_id, self.text)


# metadata = sqlalchemy.MetaData()

def startSession(conf):
    
    config = {"sqlalchemy.url":
            "mysql://"+conf["dbuser"]+":"+conf["dbpass"]+"@"+conf["dbhost"]+"/"+conf["db"]+"?charset=utf8",
            "sqlalchemy.echo":"False"}
    engine = sqlalchemy.engine_from_config(config)

    dbSession = scoped_session(
                    sessionmaker(
                        autoflush = True,
                        autocommit = False,
                        bind = engine
                    )
                )

    Base.metadata.create_all(engine)
    print ("--start DB Session--")
    return dbSession
        
"""
# テスト内容
>>> a = startSession()
--start DB Session--
"""    
