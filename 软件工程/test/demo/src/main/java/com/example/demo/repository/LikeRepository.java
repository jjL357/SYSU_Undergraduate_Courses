package com.example.demo.repository;

import com.example.demo.model.Like;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface LikeRepository extends JpaRepository<Like, Long> {

    List<Like> findByPostId(Long postId);

    Like findByUserIdAndPostId(Long userId, Long postId);
}
