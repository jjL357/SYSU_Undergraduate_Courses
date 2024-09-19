package com.example.demo.dao;
import com.example.demo.model.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public class LikeDAOImpl implements LikeDAO {

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @Override
    public void likePost(Long postId, Long userId) {
        String sql = "INSERT INTO likes (post_id, user_id) VALUES (?, ?)";
        jdbcTemplate.update(sql, postId, userId);
    }

    @Override
    public Long countLikes(Long postId) {
        String sql = "SELECT COUNT(*) FROM likes WHERE post_id = ?";
        return jdbcTemplate.queryForObject(sql, Long.class, postId);
    }

    @Override
    public List<Object[]> findTop15HotPosts() {
        String sql = "SELECT post_id, COUNT(*) AS likeCount " +
                     "FROM likes " +
                     "GROUP BY post_id " +
                     "ORDER BY likeCount DESC " +
                     "LIMIT 10";
        return jdbcTemplate.query(sql, (resultSet, i) ->
                new Object[]{resultSet.getLong("post_id"), resultSet.getLong("likeCount")});
    }

    @Override
    public List<Long>  getLikedPosts(User user) {
        Long uid = user.getUid();
        String sql = "SELECT post_id " +
                     "FROM likes " +
                     "WHERE uid = ?";
        return jdbcTemplate.queryForList(sql, Long.class, uid);
    }
}
