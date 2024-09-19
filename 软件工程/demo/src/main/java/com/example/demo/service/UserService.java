package com.example.demo.service;

import com.example.demo.model.User;

import java.util.List;

public interface UserService {

    List<User> getAllUsers();

    User saveUser(User user);

    User findUserByNameAndPassword(String name, String password);

    User getLastRegisteredUser();

    boolean isUsernameUnique(String username) ;

    public User findUserByName(String name);

    public void updateUser(User user) ;

    public User findUserByUid(Long uid) ;
        
}
