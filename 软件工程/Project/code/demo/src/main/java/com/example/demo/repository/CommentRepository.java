package com.example.demo.repository;

import com.example.demo.model.Comment;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
import java.util.Optional; // 导入 Optional 类型

public interface CommentRepository extends JpaRepository<Comment, Long> {

    List<Comment> findByPostId(Long postId);

    List<Comment> findByParentId(Long parentId);

    Optional<Comment> findById(Long id); // 将返回类型更新为 Optional<Comment>

}
