package com.example.demo.repository;

import com.example.demo.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {

    User findByNameAndPassword(String name, String password);

    User findFirstByOrderByUidDesc();

    boolean existsByName(String name);

    User findByName(String username);

    User findByUid(Long uid);
}
