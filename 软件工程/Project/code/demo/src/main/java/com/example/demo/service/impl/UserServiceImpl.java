package com.example.demo.service.impl;

import com.example.demo.model.User;
import com.example.demo.repository.UserRepository;
import com.example.demo.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class UserServiceImpl implements UserService {

    @Autowired
    private UserRepository userRepository;

    @Override
    public List<User> getAllUsers() {
        return userRepository.findAll();
    }

    @Override
    public User saveUser(User user) {
        return userRepository.save(user);
    }

    @Override
    public User findUserByNameAndPassword(String name, String password) {
        return userRepository.findByNameAndPassword(name, password);
    }

    @Override
    public User getLastRegisteredUser() {
        return userRepository.findFirstByOrderByUidDesc();
    }

    public boolean isUsernameUnique(String username) {
        User user = userRepository.findByName(username);
        return user == null;
    }

    public User findUserByName(String name) {
        return userRepository.findByName(name);
    }

    public void updateUser(User user) {
        // 更新用户信息的代码
        userRepository.save(user);
    }


    public User findUserByUid(Long uid) {
        return userRepository.findByUid(uid);
    }
}
