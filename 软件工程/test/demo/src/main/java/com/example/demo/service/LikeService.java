package com.example.demo.service;

import com.example.demo.model.Like;
import com.example.demo.repository.LikeRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
public class LikeService {

    private final LikeRepository likeRepository;

    @Autowired
    public LikeService(LikeRepository likeRepository) {
        this.likeRepository = likeRepository;
    }

    @Transactional
    public void likePost(Long userId, Long postId) {
        Like like = new Like(userId, postId);
        likeRepository.save(like);
    }

    @Transactional
    public void unlikePost(Long userId, Long postId) {
        Like like = likeRepository.findByUserIdAndPostId(userId, postId);
        if (like != null) {
            likeRepository.delete(like);
        }
    }

    public List<Like> findLikesByPostId(Long postId) {
        return likeRepository.findByPostId(postId);
    }

    public boolean isPostLikedByUser(Long userId, Long postId) {
        Like like = likeRepository.findByUserIdAndPostId(userId, postId);
        return like != null;
    }
}
