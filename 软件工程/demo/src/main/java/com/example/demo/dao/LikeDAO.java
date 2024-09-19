package com.example.demo.dao;

import java.util.List;
import com.example.demo.model.User;
public interface LikeDAO {

    void likePost(Long postId, Long userId);

    Long countLikes(Long postId);

    public List<Object[]> findTop15HotPosts();

    public List<Long> getLikedPosts(User user);
}
